from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import statsmodels.formula.api as smf


@dataclass
class LickMicrostructureConfig:
    """Parámetros estándar de microestructura de licking en ratón."""
    burst_pause_threshold_ms: float = 500.0   # estándar habitual: 500 ms o 1000 ms
    min_licks_per_burst: int = 2
    intra_burst_ili_cutoff_ms: float = 250.0  # en C57BL/6 ~90% ocurre < 160 ms
    initial_window_s: float = 60.0


class LickMicrostructureAnalyzer:
    """Extrae número de ráfagas (motivación), tamaño de ráfaga (impacto hedónico) e ILIs."""

    def __init__(self, config: Optional[LickMicrostructureConfig] = None) -> None:
        self.config = config or LickMicrostructureConfig()

    def analyze_session(self, licks_df: pd.DataFrame) -> Dict[str, Any]:
        if licks_df.empty:
            return {
                "total_licks": 0, "n_bursts": 0, "mean_burst_size": np.nan,
                "first_minute_lick_rate_hz": 0.0, "intra_burst_ili_median_ms": np.nan,
                "intra_burst_ili_cv": np.nan, "bursts_df": pd.DataFrame(),
                "all_ilis_ms": np.array([], dtype=float),
            }

        t = np.sort(licks_df["lick_timestamp_s"].to_numpy(dtype=float))
        t0 = t[0]
        ilis_ms = np.diff(t) * 1000.0
        pause_breaks = np.where(ilis_ms > self.config.burst_pause_threshold_ms)[0] + 1
        burst_slices = np.split(t, pause_breaks)

        burst_records = []
        for idx, b_times in enumerate(burst_slices):
            if len(b_times) >= self.config.min_licks_per_burst:
                b_ilis = np.diff(b_times) * 1000.0
                burst_records.append({
                    "burst_id": idx,
                    "start_s": float(b_times[0]),
                    "end_s": float(b_times[-1]),
                    "duration_s": float(b_times[-1] - b_times[0]),
                    "n_licks": int(len(b_times)),
                    "mean_ili_ms": float(np.mean(b_ilis)) if len(b_ilis) > 0 else np.nan,
                })

        bursts_df = pd.DataFrame(burst_records)
        intra_ilis = ilis_ms[ilis_ms <= self.config.intra_burst_ili_cutoff_ms]
        first_min_count = int(np.sum((t - t0) <= self.config.initial_window_s))

        return {
            "total_licks": int(len(t)),
            "n_bursts": int(len(bursts_df)),
            "mean_burst_size": float(bursts_df["n_licks"].mean()) if not bursts_df.empty else np.nan,
            "first_minute_lick_rate_hz": first_min_count / self.config.initial_window_s,
            "intra_burst_ili_median_ms": float(np.median(intra_ilis)) if len(intra_ilis) > 0 else np.nan,
            "intra_burst_ili_cv": float(np.std(intra_ilis) / np.mean(intra_ilis)) if len(intra_ilis) > 1 and np.mean(intra_ilis) > 0 else np.nan,
            "bursts_df": bursts_df,
            "all_ilis_ms": ilis_ms,
        }


class StateSpaceLearningEstimator:
    """
    Modelo espacio-estado de Smith et al. (2004) vía EM para estimar la curva de
    aprendizaje p_k con IC al 95% y el ensayo de aprendizaje IO(0.95).
    """

    def __init__(self, chance_prob: float = 0.5, max_iter: int = 150, tol: float = 1e-5) -> None:
        self.chance_prob = chance_prob
        self.max_iter = max_iter
        self.tol = tol
        self.mu = float(np.log(chance_prob / (1.0 - chance_prob)))

    def fit(self, responses: np.ndarray) -> pd.DataFrame:
        y = np.asarray(responses, dtype=float)
        K = len(y)
        if K < 5:
            raise ValueError("Se requieren al menos 5 ensayos para estimar la curva espacio-estado.")

        sigma2_e = 0.05
        x0 = 0.0

        for _ in range(self.max_iter):
            x_pred = np.zeros(K)
            v_pred = np.zeros(K)
            x_filt = np.zeros(K)
            v_filt = np.zeros(K)

            for k in range(K):
                x_p = x0 if k == 0 else x_filt[k - 1]
                v_p = sigma2_e if k == 0 else v_filt[k - 1] + sigma2_e
                x_pred[k], v_pred[k] = x_p, v_p

                xk = x_p
                for _newton in range(15):
                    pk = special.expit(self.mu + xk)
                    grad = -(xk - x_p) / v_p + (y[k] - pk)
                    hess = -1.0 / v_p - pk * (1.0 - pk)
                    step = grad / hess
                    xk -= step
                    if abs(step) < 1e-6:
                        break
                pk = special.expit(self.mu + xk)
                x_filt[k] = xk
                v_filt[k] = 1.0 / (1.0 / v_p + pk * (1.0 - pk))

            x_smooth = np.copy(x_filt)
            v_smooth = np.copy(v_filt)
            cov_lag1 = np.zeros(K)

            for k in range(K - 2, -1, -1):
                A_k = v_filt[k] / v_pred[k + 1]
                x_smooth[k] = x_filt[k] + A_k * (x_smooth[k + 1] - x_pred[k + 1])
                v_smooth[k] = v_filt[k] + (A_k ** 2) * (v_smooth[k + 1] - v_pred[k + 1])
                cov_lag1[k + 1] = A_k * v_smooth[k + 1]

            diffs = np.sum(
                (x_smooth[1:] - x_smooth[:-1]) ** 2
                + v_smooth[1:] + v_smooth[:-1] - 2.0 * cov_lag1[1:]
            )
            new_sigma2 = max((diffs + (x_smooth[0] ** 2 + v_smooth[0])) / K, 1e-4)
            if abs(new_sigma2 - sigma2_e) < self.tol:
                sigma2_e = new_sigma2
                break
            sigma2_e = new_sigma2

        sd_smooth = np.sqrt(np.maximum(v_smooth, 1e-9))
        p_mode = special.expit(self.mu + x_smooth)
        p_low95 = special.expit(self.mu + x_smooth - 1.96 * sd_smooth)
        p_high95 = special.expit(self.mu + x_smooth + 1.96 * sd_smooth)
        p_above_chance = 1.0 - stats.norm.cdf(0.0, loc=x_smooth, scale=sd_smooth)

        io_trial = np.nan
        above_95 = p_above_chance >= 0.95
        for k in range(K):
            if np.all(above_95[k:]):
                io_trial = k + 1
                break

        return pd.DataFrame({
            "trial_index": np.arange(1, K + 1),
            "observed": y,
            "p_correct_est": p_mode,
            "p_low95": p_low95,
            "p_high95": p_high95,
            "p_ideal_observer": p_above_chance,
            "io_learning_trial": io_trial,
            "sigma2_process": sigma2_e,
        })


class PsychometricChronometricAnalyzer:
    """Ajusta curvas psicométricas (tasa de acierto) y cronométricas (latencia) por condición."""

    @staticmethod
    def _logistic(x: np.ndarray, slope: float, threshold: float, lapse_low: float, lapse_high: float) -> np.ndarray:
        return lapse_low + (1.0 - lapse_low - lapse_high) * special.expit(slope * (x - threshold))

    def fit_curves(
        self,
        df: pd.DataFrame,
        stimulus_col: str,
        response_col: str = "is_hit",
        latency_col: str = "reaction_transit_ms",
        group_col: str = "opto_laser_on",
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"summary": pd.DataFrame(), "params": {}, "group_col": group_col}
        summaries = []

        for grp, sub in df.groupby(group_col):
            agg = sub.groupby(stimulus_col).agg(
                n_trials=(response_col, "count"),
                hit_rate=(response_col, "mean"),
                hit_sem=(response_col, "sem"),
                latency_median_ms=(latency_col, "median"),
                latency_iqr_ms=(
                    latency_col,
                    lambda s: np.nanpercentile(s, 75) - np.nanpercentile(s, 25) if s.notna().any() else np.nan,
                ),
            ).reset_index()
            agg[group_col] = grp
            summaries.append(agg)

            x = agg[stimulus_col].to_numpy(dtype=float)
            y = agg["hit_rate"].to_numpy(dtype=float)
            if len(x) >= 4:
                span = max(float(x.max() - x.min()), 1.0)
                p0 = [4.0 / span, float(np.median(x)), 0.05, 0.05]
                bounds = ([-5.0, x.min() - span, 0.0, 0.0], [5.0, x.max() + span, 0.49, 0.49])
                try:
                    popt, _ = optimize.curve_fit(self._logistic, x, y, p0=p0, bounds=bounds, maxfev=5000)
                    results["params"][str(grp)] = {
                        "slope": float(popt[0]),
                        "pse_threshold": float(popt[1]),
                        "lapse_low": float(popt[2]),
                        "lapse_high": float(popt[3]),
                    }
                except RuntimeError:
                    results["params"][str(grp)] = {"fit_status": "failed_convergence"}

        results["summary"] = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        return results


class HypothesisTestSelector:
    """
    Selector automático de pruebas paramétricas vs. no paramétricas, tendencias
    monotónicas (Spearman/Kendall) y modelos lineales mixtos longitudinales.
    """

    @staticmethod
    def compare_two_groups(a: np.ndarray, b: np.ndarray, paired: bool = False) -> Dict[str, Any]:
        a_clean = np.asarray(a, dtype=float)
        b_clean = np.asarray(b, dtype=float)
        a_clean = a_clean[~np.isnan(a_clean)]
        b_clean = b_clean[~np.isnan(b_clean)]

        if len(a_clean) < 3 or len(b_clean) < 3:
            return {"test": "insufficient_n", "p_value": np.nan}

        if paired:
            n_min = min(len(a_clean), len(b_clean))
            diff = a_clean[:n_min] - b_clean[:n_min]
            norm_ok = stats.shapiro(diff).pvalue > 0.05 if len(diff) <= 5000 else True
            if norm_ok:
                stat, p = stats.ttest_rel(a_clean[:n_min], b_clean[:n_min])
                return {"test": "paired_t_test", "statistic": float(stat), "p_value": float(p), "parametric": True}
            stat, p = stats.wilcoxon(a_clean[:n_min], b_clean[:n_min])
            return {"test": "wilcoxon_signed_rank", "statistic": float(stat), "p_value": float(p), "parametric": False}

        norm_a = stats.shapiro(a_clean).pvalue > 0.05 if len(a_clean) <= 5000 else True
        norm_b = stats.shapiro(b_clean).pvalue > 0.05 if len(b_clean) <= 5000 else True
        var_ok = stats.levene(a_clean, b_clean).pvalue > 0.05

        if norm_a and norm_b and var_ok:
            stat, p = stats.ttest_ind(a_clean, b_clean, equal_var=True)
            return {"test": "independent_t_test", "statistic": float(stat), "p_value": float(p), "parametric": True}
        stat, p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
        return {"test": "mann_whitney_u", "statistic": float(stat), "p_value": float(p), "parametric": False}

    @staticmethod
    def evaluate_trend(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evalúa tendencia monotónica y lineal con validación de normalidad."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_c, y_c = x_arr[mask], y_arr[mask]
        if len(x_c) < 4:
            return {"n": int(len(x_c)), "status": "insufficient_n"}
        rho, p_spear = stats.spearmanr(x_c, y_c)
        tau, p_kendall = stats.kendalltau(x_c, y_c)
        slope, intercept, r_val, p_lin, std_err = stats.linregress(x_c, y_c)
        return {
            "n": int(len(x_c)),
            "spearman_rho": float(rho),
            "spearman_p": float(p_spear),
            "kendall_tau": float(tau),
            "kendall_p": float(p_kendall),
            "linear_slope": float(slope),
            "linear_intercept": float(intercept),
            "pearson_r": float(r_val),
            "linear_p": float(p_lin),
            "linear_se": float(std_err),
        }

    @staticmethod
    def mixed_effects_trend(
        df: pd.DataFrame,
        outcome_col: str,
        fixed_terms: Tuple[str, ...] = ("session_num", "body_weight_pct", "opto_laser_on"),
        group_col: str = "mouse_id",
    ) -> Dict[str, Any]:
        """Ajusta LMM con intercepto aleatorio por ratón incluyendo solo términos con varianza."""
        valid_terms = [
            col for col in fixed_terms
            if col in df.columns and df[col].dropna().nunique() > 1
        ]
        sub = df.dropna(subset=[outcome_col, group_col] + valid_terms).copy()
        if sub[group_col].nunique() < 2 or not valid_terms or len(sub) < 10:
            return {"status": "skipped_insufficient_groups_or_variance"}
        formula = f"{outcome_col} ~ " + " + ".join(valid_terms)
        model = smf.mixedlm(formula, data=sub, groups=sub[group_col])
        fit = model.fit(reml=False)
        return {
            "status": "fitted",
            "formula": formula,
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "params": {k: float(v) for k, v in fit.params.items()},
            "pvalues": {k: float(v) for k, v in fit.pvalues.items()},
        }
