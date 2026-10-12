"""UB-ECSQ-NSGA2: NSGA-II over x = (K0, rho) (Algorithm 1).

Each candidate is built and scored by Algorithm 5:

    1. UBQ: flatten all non-bias weights and bin them into K0 equal-width bins;
       drop empty bins, leaving K~ <= K0 bins represented by their means.
    2. ECSQ merge: target K = max(1, round(rho * K~)) codewords (Eq. 12) and take
       the nearest rung of the merge ladder (exact DP merges for a lambda grid).
    3. Score: f1 = 1 - top-1 accuracy on the calibration split (one forward pass),
       f2 = 1 / CR with Huffman-coded indices (Eq. 2).

rho scales the realized number of non-empty bins K~ rather than K0, so rho = 1
means no merging. Merging is part of every evaluation; no merge step runs after
the search and no weights are retrained.
"""

import copy
import os
import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems.static import StaticProblem
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import CachedEvalSet, create_model, evaluate, get_dataset, resolve_device, setup_seed
from utils.compression import build_merge_ladder, compress, get_model_params, nearest_rung, set_model_params
from utils.compression import size_of_model
from utils.models import get_weights_preprocessing

from .display import GenerationOutput
from .encoding import NBinsRepair, decode_n_bins, gene_bounds
from .objectives import hv_bounded, objectives
from .plotting import plot_hypervolume_history, plot_pareto_front
from .wandb_utils import setup_wandb
from .warmup import find_k_bounds

BITS_PER_BYTE = 8
BYTES_PER_MB = 1024 * 1024


def decode(x, args):
    """Split one decision vector into (K0, rho)."""
    return decode_n_bins(x[0], args), float(np.clip(x[-1], 0.0, 1.0))


class GridCache:
    """UBQ statistics and merge ladder per K0.

    Both depend only on K0, and NSGA-II revisits the same K0 constantly. Only the
    O(K0) bin centroids and counts are cached; the per-weight bin assignment is
    recomputed when a model is built, so memory stays small on large networks.
    """

    def __init__(self, model, lambda_grid=None):
        self.model = model
        self.lambda_grid = lambda_grid
        self._grids = {}
        self._ladders = {}

    def grid(self, k):
        """(centers, counts) of the non-empty bins of UBQ with k bins."""
        if k not in self._grids:
            _, centers, bin_indices = compress(k, self.model, build_model=False)
            counts = np.bincount(bin_indices, minlength=len(centers)).astype(np.float64)
            self._grids[k] = (centers, counts)
        return self._grids[k]

    def bin_indices(self, k):
        _, _, bin_indices = compress(k, self.model, build_model=False)
        return bin_indices

    def ladder(self, k):
        if k not in self._ladders:
            centers, counts = self.grid(k)
            self._ladders[k] = build_merge_ladder(counts, centers, lambda_grid=self.lambda_grid)
        return self._ladders[k]

    def target_size(self, k, rho):
        """Nearest attainable merged size for (K0, rho), and its partition (Algorithm 3)."""
        centers, _ = self.grid(k)
        k_target = max(1, int(round(rho * len(centers))))
        return nearest_rung(self.ladder(k), k_target)


class ECSQRepair(NBinsRepair):
    """Canonicalize (K0, rho) so that pymoo's duplicate elimination works.

    NBinsRepair snaps the K0 gene. Many rho values reach the same merged size K
    (only sizes on the rate/distortion hull are attainable), so rho is snapped to
    K / K~ of the size it actually reaches.
    """

    def __init__(self, args, cache):
        super().__init__(args)
        self.cache = cache

    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        for i in range(len(X)):
            k, rho = decode(X[i], self.args)
            k_realized = len(self.cache.grid(k)[0])
            k_prime, _ = self.cache.target_size(k, rho)
            X[i, -1] = (k_prime / k_realized) if k_realized > 0 else 0.0
        return X


def build_merged_model(k, rung, cache, base_model):
    """A copy of `base_model` with the merged codebook of `rung` applied."""
    bin_indices = cache.bin_indices(k)
    group_of, merged_centers = rung
    merged = set_model_params(copy.deepcopy(base_model), merged_centers, group_of[bin_indices])
    merged.eval()
    return merged


def _front_columns(pf_K, rho, pf_KR, pf_KP, score_cols, pf_CR):
    """Columns of the saved fronts.

    n_bins (K0) and rho are the genes; n_bins_realized (K~) is the number of non-empty
    bins; n_parameter_sharing (K) is the number of codewords after merging; bpw is
    log2(K), the fixed-width index cost; cr is the Huffman-coded compression ratio.
    """
    data = {"n_bins": pf_K, "rho": rho}
    data["n_bins_realized"] = pf_KR.astype(int)
    data["n_parameter_sharing"] = pf_KP.astype(int)
    data["bpw"] = np.log2(np.maximum(pf_KP, 1))
    data.update(score_cols)
    data["cr"] = pf_CR
    return data


def run_ecsq(args):
    """Run UB-ECSQ-NSGA2 and save the final Pareto front to <save_dir>/final_pareto_front.csv."""
    device = resolve_device(args)
    setup_seed(args.data_seed)
    print("Objectives: f1 = 1 - calibration top-1 accuracy, f2 = 1 / CR (Huffman-coded)")
    print("Pipeline: uniform binning at K0 -> ECSQ merge to K = round(rho * K~)")

    resize_size = 256
    if args.dataset == 'imagenet':
        meta = get_weights_preprocessing(args.arch, args.weights)
        if meta is not None:
            resize_size = meta[0]
    _, val_dataset, test_dataset, num_classes, _ = get_dataset(
        args.dataset, args.data_dir, calibration_size=args.calibration_size,
        test_size=args.test_size, resize_size=resize_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = create_model(args.arch, args.dataset, num_classes, model_path=args.model_path,
                         weights=args.weights).to(device)
    model.eval()

    calibration = CachedEvalSet(val_loader, device=device)
    print(f"Calibration set cached on {device}: {len(calibration)} images")

    setup_seed(args.quantizer_seed)
    total_params = len(get_model_params(model))
    pretrained_val_acc = calibration.accuracy(model)
    total_model_size, _ = size_of_model(model)
    print(f"Pretrained calibration accuracy: {pretrained_val_acc:.4f}")
    print(f"Total parameters: {total_params}, total size (bits): {total_model_size}")
    args.pretrained_val_accuracy = pretrained_val_acc
    args.total_parameters = total_params
    args.total_model_size_bits = total_model_size

    if args.auto_k_bounds:
        args.min_bins, args.max_bins = find_k_bounds(model, calibration.accuracy, pretrained_val_acc, args)

    bins_lo, bins_hi = gene_bounds(args)
    xl = np.array([bins_lo, 0.0])
    xu = np.array([bins_hi, 1.0])
    print(f"Search bounds: gene_K in [{bins_lo:.4f}, {bins_hi:.4f}] "
          f"(K in [{args.min_bins}, {args.max_bins}]), rho in [0, 1]")

    cache = GridCache(model, lambda_grid=args.ecsq_lambda_grid)
    # A candidate's result is a deterministic function of (K0, K), so evaluate each pair once.
    fitness_cache = {}
    n_cache_hits = 0

    problem = Problem(n_var=2, n_obj=2, xl=xl, xu=xu)
    algorithm = NSGA2(
        pop_size=args.np,
        crossover=SBX(prob=args.crossover_prob, prob_var=args.crossover_prob_var, eta=args.crossover_eta),
        mutation=PM(prob=args.mutation_prob, prob_var=args.mutation_prob_var, eta=args.mutation_eta),
        repair=ECSQRepair(args, cache),
        eliminate_duplicates=True)
    # pymoo >= 0.6.2 draws from its own generator, seeded from OS entropy unless a seed is given.
    algorithm.setup(problem, termination=NoTermination(), seed=args.quantizer_seed)

    data_dir = os.path.join(args.save_dir, "steps")
    fig_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    wandb_run = setup_wandb(args)
    hv_history, evals_history, step_history = [], [], []

    output = GenerationOutput(pair_names=("K", "K'", "val", "cr"), pair_width=14)
    tqdm.write(output.header())

    next_offspring_id = 0  # unique across the whole run, to track survivors

    for step in tqdm(range(1, args.steps + 1), unit="step", disable=None):
        t0 = time.time()
        pop = algorithm.ask()
        # Mating already applies the repair; re-applying it guarantees a canonical
        # population (cheap: every lookup is cached).
        X = np.clip(pop.get("X"), xl, xu)
        X = algorithm.mating.repair._do(problem, X)
        pop.set("X", X)
        n = len(X)

        offspring_ids = np.arange(next_offspring_id, next_offspring_id + n)
        pop.set("offspring_id", offspring_ids)
        next_offspring_id += n

        F = np.zeros((n, 2))
        score_a, cr_a, bits_a = np.zeros(n), np.zeros(n), np.zeros(n)
        kreal_a, kprime_a = np.zeros(n), np.zeros(n)
        hits = 0

        for i in range(n):
            k, rho = decode(X[i], args)
            centers, _ = cache.grid(k)
            k_prime, rung = cache.target_size(k, rho)

            key = (k, k_prime)
            if key in fitness_cache:
                rec = fitness_cache[key]
                hits += 1
                n_cache_hits += 1
            else:
                merged = build_merged_model(k, rung, cache, model)
                val_score = calibration.accuracy(merged)
                size_bits, cr = size_of_model(merged)
                rec = dict(val_score=val_score, cr=cr, size_bits=size_bits,
                           k_realized=len(centers), k_prime=k_prime,
                           objectives=objectives(val_score, cr))
                del merged
                fitness_cache[key] = rec

            score_a[i], cr_a[i], bits_a[i] = rec["val_score"], rec["cr"], rec["size_bits"]
            kreal_a[i], kprime_a[i] = rec["k_realized"], rec["k_prime"]
            F[i] = rec["objectives"]

        static = StaticProblem(problem, F=F, SCORE=score_a, CR=cr_a, N_BITS=bits_a,
                               K_REAL=kreal_a, K_PRIME=kprime_a)
        Evaluator().eval(static, pop)
        algorithm.tell(infills=pop)

        # How many of this generation's offspring survived into the population / onto the front.
        this_gen_ids = set(offspring_ids.tolist())
        pop_ids = set(algorithm.pop.get("offspring_id").tolist())
        surv_pop_n = len(this_gen_ids & pop_ids)

        pf = algorithm.opt
        pf_X, pf_F, pf_SCORE, pf_CR, pf_BITS, pf_KR, pf_KP, pf_OID = pf.get(
            "X", "F", "SCORE", "CR", "N_BITS", "K_REAL", "K_PRIME", "offspring_id")
        surv_pf_n = len(this_gen_ids & set(pf_OID.tolist()))

        hv_val = hv_bounded(pf_F, args.hv_ref_point)
        hv_history.append(float(hv_val))
        evals_history.append(n - hits)
        step_history.append(step)
        pf_K = decode_n_bins(pf_X[:, 0], args)
        output.set_row(
            gen=step, evals=n - hits,
            surv_pop=(surv_pop_n, n), surv_pf=(surv_pf_n, n),
            front_size=len(pf_F),
            pair_values=[
                (pf_K.min(), pf_K.max()),
                (pf_KP.min(), pf_KP.max()),
                (pf_SCORE.min(), pf_SCORE.max(), "{:.4f}"),
                (pf_CR.min(), pf_CR.max(), "{:.1f}"),
            ],
            hv=hv_val, sec=time.time() - t0,
        )
        tqdm.write(output.row())

        pd.DataFrame({"step": step_history, "hv": hv_history, "evals": evals_history}).to_csv(
            os.path.join(args.save_dir, "hypervolume_history.csv"), index=False)

        if step % args.save_interval == 0 or step == 1 or step == args.steps:
            step_data = _front_columns(pf_K, np.clip(pf_X[:, -1], 0.0, 1.0), pf_KR, pf_KP,
                                       {"eval_score": pf_SCORE}, pf_CR)
            step_data["compressed_size_bits"] = pf_BITS
            pd.DataFrame(step_data).to_csv(os.path.join(data_dir, f"step_{step}_pareto_front.csv"))
            plot_pareto_front(
                step, pf_F, baseline=pretrained_val_acc,
                title=f"Pareto front at generation {step}/{args.steps}",
                fig_path=os.path.join(fig_dir, f"step_{step}_pareto_front.png"), wandb_run=wandb_run)
            plot_hypervolume_history(
                step_history, hv_history, fig_path=os.path.join(fig_dir, "hypervolume_history.png"),
                title=f"Hypervolume during evolution (generation {step}/{args.steps})", wandb_run=wandb_run)

        if wandb_run is not None:
            wandb_run.log({
                "iteration": step, "evolution/hv": hv_val,
                "evolution/pareto_front_size": len(pf_F),
                "evolution/ideal_eval_score": pf_SCORE.max(),
                "evolution/ideal_cr": pf_CR.max(),
                "evolution/cache_hits": n_cache_hits}, step=step)

    plot_pareto_front(
        args.steps, pf_F, baseline=pretrained_val_acc, title="Final Pareto front (calibration split)",
        fig_path=os.path.join(fig_dir, "final_pareto_front.png"), final=True, wandb_run=wandb_run)

    test_scores = np.full(len(pf_X), np.nan)
    if args.eval_test:
        test_scores = _evaluate_front_on_test(pf_X, pf_KP, cache, model, test_dataset, args, device)

    data = _front_columns(pf_K, np.clip(pf_X[:, -1], 0.0, 1.0), pf_KR, pf_KP,
                          {"val_eval_score": pf_SCORE, "test_eval_score": test_scores}, pf_CR)
    data["compressed_size_bits"] = pf_BITS
    data["compressed_size_bytes"] = pf_BITS / BITS_PER_BYTE
    data["compressed_size_mb"] = pf_BITS / (BITS_PER_BYTE * BYTES_PER_MB)
    df = pd.DataFrame(data).sort_values("cr").reset_index(drop=True)

    csv_path = os.path.join(args.save_dir, "final_pareto_front.csv")
    df.to_csv(csv_path)
    print(f"\nSaved Pareto front to: {csv_path}")
    print(f"Distinct (K0, K) evaluated: {len(fitness_cache)}, cache hits: {n_cache_hits}")
    columns = ["n_bins", "rho", "n_parameter_sharing", "bpw", "val_eval_score", "cr"]
    if args.eval_test:
        columns.insert(5, "test_eval_score")
    print(df[columns].to_string(index=False))

    if wandb_run is not None:
        import wandb
        wandb_run.log({"tables/final_pareto_front": wandb.Table(dataframe=df)}, step=args.steps)
        wandb_run.finish()

    return df


def _evaluate_front_on_test(pf_X, pf_KP, cache, model, test_dataset, args, device):
    """Top-1 accuracy of every Pareto solution on the held-out test split."""
    print(f"\nEvaluating {len(pf_X)} Pareto solutions on the test split ({len(test_dataset)} images)...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    scores = np.zeros(len(pf_X))
    for i, x in enumerate(tqdm(pf_X, disable=None)):
        k, rho = decode(x, args)
        k_prime, rung = cache.target_size(k, rho)
        assert k_prime == int(pf_KP[i]), "Pareto solution does not rebuild to its evaluated codebook"
        merged = build_merged_model(k, rung, cache, model)
        scores[i] = evaluate(merged, test_loader, device=device)
        del merged
    return scores
