"""Optional Weights & Biases logging (enabled with --wandb)."""


def setup_wandb(args):
    """Start a W&B run when --wandb is set; returns None otherwise."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("--wandb needs the wandb package (pip install wandb)") from exc

    run_name = args.wandb_name or f"{args.dataset}_{args.arch}_seed{args.quantizer_seed}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        dir=args.save_dir,
        tags=[args.dataset, args.arch],
    )
