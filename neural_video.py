from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class PhotometryPETHAnalyzer:
    """
    Preprocesamiento isosbéstico (405 nm) y estadística peri-evento punto a punto
    con IC por bootstrap, permutaciones y umbral consecutivo (FiPhoPHA / Jean-Richard-dit-Bressel).
    Compatible también con trazas dF/F ya extraídas de imagenología de calcio celular.
    """

    def __init__(self, sampling_rate_hz: float = 100.0, consecutive_threshold: int = 4) -> None:
        self.fs = sampling_rate_hz
        self.consecutive_threshold = consecutive_threshold

    @staticmethod
    def compute_dff(signal_exp: np.ndarray, signal_iso: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(signal_iso, signal_exp, 1)
        fitted_iso = slope * signal_iso + intercept
        return (signal_exp - fitted_iso) / np.where(fitted_iso == 0, np.nan, fitted_iso)

    def extract_peri_event_zscore(
        self,
        timestamps_s: np.ndarray,
        dff: np.ndarray,
        event_times_s: np.ndarray,
        window_s: Tuple[float, float] = (-2.0, 4.0),
        baseline_s: Tuple[float, float] = (-2.0, -0.5),
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = int(round((window_s[1] - window_s[0]) * self.fs)) + 1
        rel_time = np.linspace(window_s[0], window_s[1], n_samples)
        base_mask = (rel_time >= baseline_s[0]) & (rel_time <= baseline_s[1])

        traces = []
        for ev in event_times_s:
            if np.isnan(ev):
                continue
            target_t = ev + rel_time
            if target_t[0] < timestamps_s[0] or target_t[-1] > timestamps_s[-1]:
                continue
            interp_dff = np.interp(target_t, timestamps_s, dff)
            mu_b = np.nanmean(interp_dff[base_mask])
            sd_b = np.nanstd(interp_dff[base_mask])
            traces.append((interp_dff - mu_b) / (sd_b if sd_b > 1e-9 else 1.0))

        if not traces:
            return rel_time, np.empty((0, n_samples), dtype=float)
        return rel_time, np.asarray(traces, dtype=float)

    def _apply_consecutive_threshold(self, bool_vec: np.ndarray) -> np.ndarray:
        out = np.zeros_like(bool_vec, dtype=bool)
        k = self.consecutive_threshold
        run = 0
        for idx, val in enumerate(bool_vec):
            run = run + 1 if val else 0
            if run >= k:
                out[idx - k + 1 : idx + 1] = True
        return out

    def bootstrap_ci_vs_baseline(
        self,
        traces: np.ndarray,
        n_boot: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        if traces.ndim != 2 or traces.shape[0] < 2:
            raise ValueError("Se requieren al menos 2 trazas válidas (matriz 2D) para calcular bootstrap.")
        rng = np.random.default_rng(seed)
        n_trials, n_time = traces.shape
        boot_means = np.empty((n_boot, n_time), dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n_trials, size=n_trials)
            boot_means[b] = np.nanmean(traces[idx], axis=0)

        alpha = (1.0 - ci_level) * 100.0 / 2.0
        ci_low = np.percentile(boot_means, alpha, axis=0)
        ci_high = np.percentile(boot_means, 100.0 - alpha, axis=0)
        raw_sig = (ci_low > 0.0) | (ci_high < 0.0)

        return {
            "mean": np.nanmean(traces, axis=0),
            "sem": stats.sem(traces, axis=0, nan_policy="omit"),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "sig_mask": self._apply_consecutive_threshold(raw_sig),
        }

    def permutation_test_two_conditions(
        self,
        traces_a: np.ndarray,
        traces_b: np.ndarray,
        n_perm: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        obs_diff = np.nanmean(traces_a, axis=0) - np.nanmean(traces_b, axis=0)
        pooled = np.vstack([traces_a, traces_b])
        n_a = len(traces_a)

        null_diffs = np.empty((n_perm, pooled.shape[1]), dtype=float)
        for p in range(n_perm):
            perm_idx = rng.permutation(len(pooled))
            null_diffs[p] = (
                np.nanmean(pooled[perm_idx[:n_a]], axis=0)
                - np.nanmean(pooled[perm_idx[n_a:]], axis=0)
            )

        p_vals = np.mean(np.abs(null_diffs) >= np.abs(obs_diff), axis=0)
        return {
            "obs_diff": obs_diff,
            "p_values": p_vals,
            "sig_mask": self._apply_consecutive_threshold(p_vals < alpha),
        }


class VideoKinematicEncodingAnalyzer:
    """
    Procesa seguimiento de pose (DeepLabCut/SLEAP/ndx-pose), calcula métricas
    en arena rectangular-triangular, evalúa sílabas MoSeq y cuantifica varianza
    neural única mediante Regresión Ridge (Musall et al., 2019).
    """

    def __init__(self, vertex_xy_cm: Tuple[float, float], pixels_per_cm: float = 10.0) -> None:
        self.vertex_xy_cm = np.array(vertex_xy_cm, dtype=float)
        self.px_per_cm = pixels_per_cm

    def compute_kinematics(self, pose_df: pd.DataFrame, likelihood_thresh: float = 0.9) -> pd.DataFrame:
        df = pose_df.copy().sort_values("timestamp_s").reset_index(drop=True)
        for part in ["nose", "midback", "left_ear", "right_ear"]:
            if f"{part}_x" not in df.columns or f"{part}_y" not in df.columns:
                continue
            lik_col = f"{part}_likelihood"
            if lik_col in df.columns:
                bad = df[lik_col] < likelihood_thresh
                df.loc[bad, [f"{part}_x", f"{part}_y"]] = np.nan
            df[f"{part}_x"] = df[f"{part}_x"].interpolate(limit_direction="both") / self.px_per_cm
            df[f"{part}_y"] = df[f"{part}_y"].interpolate(limit_direction="both") / self.px_per_cm

        dt = np.gradient(df["timestamp_s"].to_numpy(dtype=float))
        pos_dt = dt[dt > 0]
        dt = np.where(dt <= 0, np.nanmedian(pos_dt) if len(pos_dt) else 1.0 / 30.0, dt)

        nose_xy = df[["nose_x", "nose_y"]].to_numpy()
        mid_xy = df[["midback_x", "midback_y"]].to_numpy()
        df["dist_to_vertex_cm"] = np.linalg.norm(nose_xy - self.vertex_xy_cm, axis=1)

        vx = np.gradient(mid_xy[:, 0]) / dt
        vy = np.gradient(mid_xy[:, 1]) / dt
        df["body_speed_cm_s"] = np.hypot(vx, vy)

        head_vec = nose_xy - mid_xy
        target_vec = self.vertex_xy_cm - mid_xy
        head_angle = np.arctan2(head_vec[:, 1], head_vec[:, 0])
        target_angle = np.arctan2(target_vec[:, 1], target_vec[:, 0])
        df["head_heading_rad"] = np.unwrap(head_angle)
        df["angular_velocity_rad_s"] = np.gradient(df["head_heading_rad"]) / dt
        df["vertex_alignment_err_deg"] = np.degrees(
            np.arctan2(np.sin(head_angle - target_angle), np.cos(head_angle - target_angle))
        )
        return df

    @staticmethod
    def syllable_group_comparison(syllable_df: pd.DataFrame, group_col: str = "group_condition") -> pd.DataFrame:
        """Compara uso de sílabas conductuales (Keypoint-MoSeq) entre grupos con Kruskal-Wallis + FDR."""
        rows = []
        for syl, sub in syllable_df.groupby("syllable_id"):
            groups = [g["usage_freq"].to_numpy() for _, g in sub.groupby(group_col)]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                stat, p = stats.kruskal(*groups)
                rows.append({"syllable_id": syl, "kruskal_H": float(stat), "p_value": float(p)})
        res = pd.DataFrame(rows)
        if not res.empty:
            res["p_fdr"] = stats.false_discovery_control(res["p_value"].to_numpy())
        return res

    @staticmethod
    def ridge_unique_variance(
        neural_y: np.ndarray,
        predictor_groups: Dict[str, np.ndarray],
        n_folds: int = 5,
        alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    ) -> Dict[str, float]:
        """Calcula R^2 completo y varianza única (delta R^2) por dominio con validación cruzada sin barajar."""
        X_full = np.hstack(list(predictor_groups.values()))
        kf = KFold(n_splits=n_folds, shuffle=False)

        def _cv_r2(X: np.ndarray) -> float:
            preds = np.zeros_like(neural_y, dtype=float)
            for tr, te in kf.split(X):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr])
                X_te = scaler.transform(X[te])
                model = RidgeCV(alphas=alphas).fit(X_tr, neural_y[tr])
                preds[te] = model.predict(X_te)
            ss_res = np.sum((neural_y - preds) ** 2)
            ss_tot = np.sum((neural_y - np.mean(neural_y)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        r2_full = _cv_r2(X_full)
        out = {"R2_full": r2_full}
        if len(predictor_groups) > 1:
            for name in predictor_groups:
                X_red = np.hstack([arr for k, arr in predictor_groups.items() if k != name])
                out[f"delta_R2_{name}"] = r2_full - _cv_r2(X_red)
        return out
