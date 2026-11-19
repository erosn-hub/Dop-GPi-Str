from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .behavior_stats import (
    HypothesisTestSelector,
    LickMicrostructureAnalyzer,
    PsychometricChronometricAnalyzer,
    StateSpaceLearningEstimator,
)
from .core import ExperimentMetadata, TASK_REGISTRY
from .neural_video import PhotometryPETHAnalyzer, VideoKinematicEncodingAnalyzer


class ExactNeuroPlotter:
    """Genera gráficos exactos de publicación con bandas de confianza y marcas estadísticas."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    def plot_learning_and_psychometrics(
        self,
        ss_df: pd.DataFrame,
        psy_res: Dict[str, Any],
        stimulus_col: str,
        filename: str = "fig1_learning_psychometrics.png",
    ) -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=300)

        ax = axes[0]
        ax.fill_between(ss_df["trial_index"], ss_df["p_low95"], ss_df["p_high95"], alpha=0.25, color="#1f77b4", label="IC 95%")
        ax.plot(ss_df["trial_index"], ss_df["p_correct_est"], color="#1f77b4", lw=2, label="p(acierto) estimada")
        ax.axhline(0.5, ls="--", color="gray", lw=1, label="Azar")
        io_t = ss_df["io_learning_trial"].iloc[0]
        if not np.isnan(io_t):
            ax.axvline(io_t, ls=":", color="#d62728", lw=1.8, label=f"IO(0.95) ensayo {int(io_t)}")
        ax.set(title="Dinámica de Aprendizaje (Espacio-Estado)", xlabel="Ensayo", ylabel="Probabilidad de acierto", ylim=(0, 1.05))
        ax.legend(frameon=False, fontsize=8)

        grp_col = psy_res["group_col"]
        sum_df = psy_res["summary"]
        ax = axes[1]
        for grp, sub in sum_df.groupby(grp_col):
            ax.errorbar(sub[stimulus_col], sub["hit_rate"], yerr=sub["hit_sem"], marker="o", capsize=3, label=f"{grp_col}={grp}")
        ax.set(title="Curva Psicométrica", xlabel=stimulus_col, ylabel="Tasa de acierto (Hit)", ylim=(0, 1.05))
        ax.legend(frameon=False, fontsize=8)

        ax = axes[2]
        for grp, sub in sum_df.groupby(grp_col):
            ax.plot(sub[stimulus_col], sub["latency_median_ms"], marker="s", label=f"{grp_col}={grp}")
        ax.set(title="Curva Cronométrica (Latencia al 1er Lick)", xlabel=stimulus_col, ylabel="Mediana latencia (ms)")
        ax.legend(frameon=False, fontsize=8)

        sns.despine()
        plt.tight_layout()
        out = self.out_dir / filename
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_peth_with_significance(
        self,
        rel_time: np.ndarray,
        res_a: Dict[str, np.ndarray],
        res_b: Optional[Dict[str, np.ndarray]] = None,
        perm_res: Optional[Dict[str, np.ndarray]] = None,
        labels: Tuple[str, str] = ("Hits / Laser OFF", "No-Hit / Laser ON"),
        title: str = "PETH Neural (IC Bootstrap + Umbral Consecutivo)",
        filename: str = "fig2_photometry_peth.png",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        ax.axvline(0.0, ls="--", color="black", lw=1.2)
        ax.axhline(0.0, ls=":", color="gray", lw=1.0)

        ax.fill_between(rel_time, res_a["ci_low"], res_a["ci_high"], alpha=0.2, color="#2ca02c")
        ax.plot(rel_time, res_a["mean"], color="#2ca02c", lw=2, label=labels[0])

        y_bar = float(np.nanmax(res_a["ci_high"]))
        if res_b is not None:
            ax.fill_between(rel_time, res_b["ci_low"], res_b["ci_high"], alpha=0.2, color="#ff7f0e")
            ax.plot(rel_time, res_b["mean"], color="#ff7f0e", lw=2, label=labels[1])
            y_bar = max(y_bar, float(np.nanmax(res_b["ci_high"])))
        y_bar = y_bar + 0.15 * max(abs(y_bar), 1.0)

        if np.any(res_a["sig_mask"]):
            ax.plot(rel_time[res_a["sig_mask"]], np.full(np.sum(res_a["sig_mask"]), y_bar), ".", color="#2ca02c", markersize=4, label="Sig. vs base (A)")
        if perm_res is not None and np.any(perm_res["sig_mask"]):
            ax.plot(rel_time[perm_res["sig_mask"]], np.full(np.sum(perm_res["sig_mask"]), y_bar * 1.08), ".", color="#d62728", markersize=5, label="Permutación A vs B")

        ax.set(title=title, xlabel="Tiempo desde evento (s)", ylabel="z-score ΔF/F")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        sns.despine()
        plt.tight_layout()
        out = self.out_dir / filename
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_correlation_and_encoding(
        self,
        trial_table: pd.DataFrame,
        ridge_res: Optional[Dict[str, float]] = None,
        ilis_ms: Optional[np.ndarray] = None,
        filename: str = "fig3_correlations_microstructure_ridge.png",
    ) -> Path:
        n_panels = 2 if ridge_res is None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.6), dpi=300)

        num_cols = [c for c in ["vertex_hold_ms", "hold_target_ms", "reaction_transit_ms", "reward_delivered_ul", "body_weight_pct", "is_hit", "mean_speed_cm_s", "mean_neural_z"] if c in trial_table.columns]
        num_df = trial_table[num_cols].dropna(axis=1, how="all")
        corr = num_df.corr(method="spearman")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={"label": "Spearman ρ"})
        axes[0].set_title("Correlaciones Conducta–Peso–Video–Neural")

        ax_ili = axes[1]
        if ilis_ms is not None and len(ilis_ms) > 0:
            clipped = ilis_ms[ilis_ms <= 1000.0]
            ax_ili.hist(clipped, bins=40, color="#17becf", edgecolor="white")
            ax_ili.axvline(160.0, ls="--", color="#d62728", lw=1.2, label="160 ms ref. C57BL/6")
            ax_ili.axvline(500.0, ls=":", color="black", lw=1.2, label="500 ms corte ráfaga")
            ax_ili.set(title="Distribución de Intervalos Inter-Lick (ILI)", xlabel="ILI (ms)", ylabel="Conteo")
            ax_ili.legend(frameon=False, fontsize=8)
        else:
            ax_ili.text(0.5, 0.5, "Sin datos de ILI", ha="center", va="center")

        if ridge_res is not None:
            ax_r = axes[2]
            keys = list(ridge_res.keys())
            vals = [ridge_res[k] for k in keys]
            ax_r.bar(keys, vals, color=["#34495e" if k == "R2_full" else "#3498db" for k in keys])
            ax_r.set(title="Modelo Ridge: Varianza Total y Única (ΔR²)", ylabel="R² / ΔR² (CV en bloques)")
            ax_r.tick_params(axis="x", rotation=25)

        sns.despine()
        plt.tight_layout()
        out = self.out_dir / filename
        fig.savefig(out)
        plt.close(fig)
        return out


def run_complete_pipeline(
    task_name: str,
    metadata: ExperimentMetadata,
    trials_df: pd.DataFrame,
    licks_df: pd.DataFrame,
    photometry_df: Optional[pd.DataFrame] = None,
    pose_df: Optional[pd.DataFrame] = None,
    output_dir: Path = Path("outputs_neuroflex"),
) -> Dict[str, Any]:
    """Orquestador completo de tendencias, correlaciones, PETH, video y gráficos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_json(output_dir / "experiment_metadata_audit.json")

    task = TASK_REGISTRY[task_name](metadata)
    clean_trials = task.validate_and_preprocess(trials_df)

    lick_analyzer = LickMicrostructureAnalyzer()
    lick_stats = lick_analyzer.analyze_session(licks_df)

    ss_df = StateSpaceLearningEstimator(chance_prob=0.5).fit(clean_trials["is_hit"].to_numpy())
    psy_res = PsychometricChronometricAnalyzer().fit_curves(clean_trials, stimulus_col=task.primary_stimulus_column())

    # Tendencias longitudinales y comparaciones pareadas/independientes automáticas
    trend_hit = HypothesisTestSelector.evaluate_trend(clean_trials.index.to_numpy(), clean_trials["is_hit"].to_numpy())
    trend_lat = HypothesisTestSelector.evaluate_trend(clean_trials.index.to_numpy(), clean_trials["reaction_transit_ms"].to_numpy())
    lmm_lat = HypothesisTestSelector.mixed_effects_trend(clean_trials, outcome_col="reaction_transit_ms")

    opto_comp = None
    if clean_trials["opto_laser_on"].nunique() == 2:
        lat_off = clean_trials.loc[clean_trials["opto_laser_on"] == 0, "reaction_transit_ms"].to_numpy()
        lat_on = clean_trials.loc[clean_trials["opto_laser_on"] == 1, "reaction_transit_ms"].to_numpy()
        opto_comp = HypothesisTestSelector.compare_two_groups(lat_off, lat_on, paired=False)

    plotter = ExactNeuroPlotter(output_dir / "figures")
    figures = [str(plotter.plot_learning_and_psychometrics(ss_df, psy_res, stimulus_col=task.primary_stimulus_column()))]

    ridge_res = None
    dff = None
    if photometry_df is not None and not photometry_df.empty:
        peth = PhotometryPETHAnalyzer(sampling_rate_hz=metadata.sampling_rate_hz)
        if "signal_465" in photometry_df.columns:
            if "isosbestic_405" in photometry_df.columns and photometry_df["isosbestic_405"].notna().all():
                dff = peth.compute_dff(photometry_df["signal_465"].to_numpy(), photometry_df["isosbestic_405"].to_numpy())
            else:
                dff = photometry_df["signal_465"].to_numpy()
        elif "dff" in photometry_df.columns:
            dff = photometry_df["dff"].to_numpy()
        else:
            raise ValueError("photometry_df debe contener 'signal_465' o 'dff'.")

        t_neural = photometry_df["timestamp_s"].to_numpy()

        # 1) PETH alineado al Tono Auditivo (Hits vs Omisiones con tono válido)
        hits_tone = clean_trials.loc[clean_trials["is_hit"] == 1, "tone_onset_s"].dropna().to_numpy()
        miss_tone = clean_trials.loc[clean_trials["is_hit"] == 0, "tone_onset_s"].dropna().to_numpy()
        if len(hits_tone) >= 2:
            rel_t, z_hits = peth.extract_peri_event_zscore(t_neural, dff, hits_tone)
            if len(z_hits) >= 2:
                res_hits = peth.bootstrap_ci_vs_baseline(z_hits)
                res_miss, perm_res = None, None
                if len(miss_tone) >= 5:
                    _, z_miss = peth.extract_peri_event_zscore(t_neural, dff, miss_tone)
                    if len(z_miss) >= 5:
                        res_miss = peth.bootstrap_ci_vs_baseline(z_miss)
                        perm_res = peth.permutation_test_two_conditions(z_hits, z_miss)
                figures.append(str(plotter.plot_peth_with_significance(
                    rel_t, res_hits, res_miss, perm_res,
                    labels=("Hits (Tono)", "Omisión (Tono)"),
                    title="PETH Neural Alineado al Tono Auditivo",
                    filename="fig2a_photometry_peth_tone.png",
                )))

        # 2) PETH alineado a Entrada al Vértice Triangular (Hits vs Salidas Prematuras / No-Hit)
        hits_vtx = clean_trials.loc[clean_trials["is_hit"] == 1, "vertex_enter_s"].dropna().to_numpy()
        miss_vtx = clean_trials.loc[clean_trials["is_hit"] == 0, "vertex_enter_s"].dropna().to_numpy()
        if len(hits_vtx) >= 2:
            rel_v, zv_hits = peth.extract_peri_event_zscore(t_neural, dff, hits_vtx)
            if len(zv_hits) >= 2:
                res_vhits = peth.bootstrap_ci_vs_baseline(zv_hits)
                res_vmiss, perm_vres = None, None
                if len(miss_vtx) >= 5:
                    _, zv_miss = peth.extract_peri_event_zscore(t_neural, dff, miss_vtx)
                    if len(zv_miss) >= 5:
                        res_vmiss = peth.bootstrap_ci_vs_baseline(zv_miss)
                        perm_vres = peth.permutation_test_two_conditions(zv_hits, zv_miss)
                figures.append(str(plotter.plot_peth_with_significance(
                    rel_v, res_vhits, res_vmiss, perm_vres,
                    labels=("Hits (Vértice)", "Salida Prematura / No-Hit (Vértice)"),
                    title="PETH Neural Alineado a Entrada al Vértice Triangular",
                    filename="fig2b_photometry_peth_vertex.png",
                )))

        # Promedio neural post-tono por ensayo para correlaciones multivariadas
        clean_trials["mean_neural_z"] = [
            float(np.nanmean(dff[(t_neural >= ts) & (t_neural <= ts + 1.0)])) if not np.isnan(ts) else np.nan
            for ts in clean_trials["tone_onset_s"]
        ]

    if pose_df is not None and not pose_df.empty:
        vid = VideoKinematicEncodingAnalyzer(metadata.arena.triangle_vertex_xy_cm, metadata.arena.pixels_per_cm)
        kin_df = vid.compute_kinematics(pose_df)
        t_vid = kin_df["timestamp_s"].to_numpy()
        spd = kin_df["body_speed_cm_s"].to_numpy()
        clean_trials["mean_speed_cm_s"] = [
            float(np.nanmean(spd[(t_vid >= ts) & (t_vid <= ts + 2.0)])) if not np.isnan(ts) else np.nan
            for ts in clean_trials["trial_start_s"]
        ]
        if dff is not None and len(kin_df) >= 25:
            y_neural = np.interp(t_vid, photometry_df["timestamp_s"].to_numpy(), dff)
            kin_mat = kin_df[["body_speed_cm_s", "angular_velocity_rad_s", "dist_to_vertex_cm"]].to_numpy()
            cue_pulse = np.zeros((len(t_vid), 1))
            for ts in clean_trials["tone_onset_s"].dropna():
                cue_pulse[(t_vid >= ts) & (t_vid <= ts + 0.5), 0] = 1.0
            ridge_res = vid.ridge_unique_variance(y_neural, {"cinematica": kin_mat, "tono": cue_pulse})

    figures.append(str(plotter.plot_correlation_and_encoding(clean_trials, ridge_res=ridge_res, ilis_ms=lick_stats["all_ilis_ms"])))

    return {
        "n_trials": int(len(clean_trials)),
        "overall_hit_rate": float(clean_trials["is_hit"].mean()),
        "io_learning_trial": float(ss_df["io_learning_trial"].iloc[0]),
        "trend_hit": trend_hit,
        "trend_latency": trend_lat,
        "lmm_latency": lmm_lat,
        "opto_latency_comparison": opto_comp,
        "lick_microstructure": {k: v for k, v in lick_stats.items() if not isinstance(v, (pd.DataFrame, np.ndarray))},
        "psychometrics": psy_res["params"],
        "ridge_variance": ridge_res,
        "figures": figures,
    }
