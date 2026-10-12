"""Per-generation summary table printed during the search."""

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output


class PairColumn(Column):
    """A column holding a pre-formatted "lo/hi" string."""

    def set_pair(self, lo, hi, fmt="{:.0f}"):
        self.set(f"{fmt.format(lo)}/{fmt.format(hi)}")


class FormattedColumn(Column):
    """A single-value column that takes a pre-formatted string (no scientific notation)."""

    def set_value(self, value, fmt="{:.4f}"):
        self.set(fmt.format(value) if value is not None else None)


class GenerationOutput(Output):
    """gen | evals | surv_pop | surv_pf | front_size | <ideal/nadir pairs> | HV | sec.

    evals: new evaluations this generation (cache hits excluded).
    surv_pop / surv_pf: of this generation's offspring, how many survived into the
    next population / landed on the current Pareto front.
    The pair columns show the minimum/maximum of each quantity over the front.
    """

    _SEP = " "

    def __init__(self, pair_names=("K", "K'", "val", "cr"), pair_width=14):
        super().__init__()
        self.gen = Column("gen", width=4)
        self.evals = Column("evals", width=6)
        self.surv_pop = PairColumn("surv_pop", width=11)
        self.surv_pf = PairColumn("surv_pf", width=11)
        self.front_size = Column("front_size", width=10)
        self.lead_columns = [self.gen, self.evals, self.surv_pop, self.surv_pf, self.front_size]
        self.pairs = [PairColumn(name, width=pair_width) for name in pair_names]
        self.hv = FormattedColumn("HV", width=8)
        self.sec = FormattedColumn("sec", width=7)
        self.columns = [*self.lead_columns, *self.pairs, self.hv, self.sec]

    def header(self, border=False):
        cols = self._SEP.join(col.header() for col in self.columns)
        pair_start = sum(c.width + len(self._SEP) for c in self.lead_columns)
        pair_width = sum(c.width for c in self.pairs) + len(self._SEP) * (len(self.pairs) - 1)
        title = " " * pair_start + "ideal / nadir over the front".center(pair_width)
        return "\n".join([title, cols, "-" * len(cols)])

    def text(self):
        return self._SEP.join(col.text() for col in self.columns)

    def set_row(self, gen, evals, surv_pop, surv_pf, front_size, pair_values, hv, sec):
        """surv_pop, surv_pf: (count, n). pair_values: one (lo, hi[, fmt]) per pair column."""
        self.gen.set(gen)
        self.evals.set(evals)
        self.surv_pop.set_pair(*surv_pop)
        self.surv_pf.set_pair(*surv_pf)
        self.front_size.set(front_size)
        for col, spec in zip(self.pairs, pair_values):
            fmt = spec[2] if len(spec) > 2 else "{:.0f}"
            col.set_pair(spec[0], spec[1], fmt)
        self.hv.set_value(hv, "{:.4f}")
        self.sec.set_value(sec, "{:.1f}")

    def row(self):
        return self.text()
