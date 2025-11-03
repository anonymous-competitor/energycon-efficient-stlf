import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, f, rankdata, studentized_range
from scipy.stats import f as f_dist


def seed_agg(df, metric):
    g = df.groupby(["consumer_id", "horizon", "method"], as_index=False)[metric].mean()
    return g


# 2) Build paired deltas per consumer for a horizon
def deltas_for_horizon(g, h, metric, base, prop="fine_tuned"):
    t = (
        g[g["horizon"] == h]
        .pivot(index="consumer_id", columns="method", values=metric)[[prop, base]]
        .dropna()
    )
    return (t[prop] - t[base]).to_numpy()


# 3) Wilcoxon + HL effect and a simple nonparametric CI via Walsh averages
def hl_and_ci(d, alpha=0.05):
    # HL = median of Walsh averages
    w = (d[:, None] + d[None, :]).ravel() / 2.0
    w.sort()
    hl = np.median(w)
    # nonparametric CI for the median-of-paired differences (order-statistic band)
    # simple large-sample sign CI around the sample median of d:
    d_sorted = np.sort(d)
    n = len(d_sorted)
    k = int(np.ceil((n / 2.0) - 0.98 * np.sqrt(n)))
    k = max(0, min(k, n - 1))
    ci = (d_sorted[k], d_sorted[-k - 1])
    return float(hl), (float(ci[0]), float(ci[1]))


def wilcoxon_blocked(d):
    # Signed-rank test on paired deltas
    # H0: median(d)=0 ; H1 (two choices):
    #  - 'less'  -> proposed is better (Δ<0)
    #  - 'greater'-> proposed is worse  (Δ>0)
    d = np.asarray(d, dtype=float)
    out_less = wilcoxon(d, alternative="less", zero_method="wilcox")
    out_greater = wilcoxon(d, alternative="greater", zero_method="wilcox")
    return {"p_less": out_less.pvalue, "p_greater": out_greater.pvalue}


# 4) Friedman + Nemenyi (CD) on consumer ranks across methods
# in friedman_block:
def friedman_block(g, h, metric):
    T = (
        g[g["horizon"] == h]
        .pivot(index="consumer_id", columns="method", values=metric)
        .dropna()
    )
    N, k = T.shape
    if N < 2 or k < 2:
        return {
            "row_type": "friedman",
            "metric": metric,
            "n_consumers": int(N),
            "p_friedman": np.nan,
            "avg_ranks": None,
            "CD_0.05": np.nan,
            "baseline": "ALL",
        }
    # rank per consumer (lower=better)
    R = T.apply(
        lambda r: rankdata(r.values, method="average"), axis=1, result_type="expand"
    ).to_numpy()
    Rbar = R.mean(axis=0)
    # Friedman chi-square
    chi2 = (12 * N) / (k * (k + 1)) * (np.sum(Rbar**2) - k * (k + 1) ** 2 / 4.0)
    # Iman–Davenport F
    F_ID = ((N - 1) * chi2) / (N * (k - 1) - chi2)
    # p-value
    p_ID = 1.0 - f_dist.cdf(F_ID, dfn=k - 1, dfd=(k - 1) * (N - 1))
    # Nemenyi critical difference at α=0.05
    q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0)
    CD = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))
    return {
        "row_type": "friedman",
        "n_consumers": int(N),
        "p_friedman": float(p_ID),
        "avg_ranks": dict(zip(T.columns.tolist(), Rbar)),
        "CD_0.05": CD,
        "baseline": "ALL",
        "metric": metric,
    }
