"""
Script ejecutable de demostración para neuroflex_stats.
Simula una cohorte de ratones en la arena rectangular-triangular con 3 inputs:
  1) Permanencia modulada (X ms) en vértice triangular
  2) Estímulo auditivo (tono)
  3) Licómetro en pared contraria (2-3 uL) + registro de fotometría y pose de video.

Ejecución:
    python run_example.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from neuroflex_stats import (
    ArenaGeometry,
    ExperimentMetadata,
    SessionPhysiology,
    run_complete_pipeline,
)


def generate_synthetic_dataset(seed: int = 2026):
    rng = np.random.default_rng(seed)
    mice = ["M01", "M02", "M03", "M04"]
    sessions = ["S01", "S02", "S03"]
    hold_targets = [300.0, 500.0, 700.0, 900.0]

    phys_list = []
    trial_rows = []
    lick_rows = []
    trial_counter = 1
    current_t = 5.0

    for m_idx, mouse in enumerate(mice):
        base_w = 24.5 + m_idx * 0.6
        for s_idx, sess in enumerate(sessions):
            curr_w = base_w * (0.805 + rng.normal(0, 0.008))
            phys_list.append(
                SessionPhysiology(
                    session_id=sess,
                    mouse_id=mouse,
                    session_date=f"2026-11-{10 + s_idx * 2:02d}",
                    baseline_weight_g=round(base_w, 2),
                    current_weight_g=round(curr_w, 2),
                    hours_since_last_session=48.0,
                    supplemental_water_ml=round(float(rng.uniform(0.4, 0.8)), 2),
                    group_condition="GPi_ChR2" if m_idx % 2 == 0 else "Control_eYFP",
                )
            )

            for t_in_sess in range(30):
                target_ms = float(rng.choice(hold_targets))
                opto_on = int(t_in_sess >= 15)
                skill = -0.3 + 0.85 * s_idx + 0.02 * t_in_sess - 0.0015 * (target_ms - 500.0) - 0.35 * opto_on
                p_hit = 1.0 / (1.0 + np.exp(-skill))
                is_hit = rng.random() < p_hit

                v_enter = current_t + 1.0
                if is_hit:
                    actual_hold = target_ms + rng.uniform(15.0, 80.0)
                    tone_on = v_enter + actual_hold / 1000.0
                    transit_s = rng.lognormal(mean=-0.35 + 0.15 * opto_on, sigma=0.25)
                    first_lick = tone_on + transit_s
                    reward_ul = float(rng.uniform(2.0, 3.0))
                    outcome = "hit"

                    n_licks = int(rng.integers(8, 22))
                    lick_t = first_lick
                    for l_i in range(n_licks):
                        lick_rows.append({
                            "session_id": sess,
                            "mouse_id": mouse,
                            "trial_id": trial_counter,
                            "lick_timestamp_s": lick_t,
                            "lick_duration_ms": float(rng.uniform(35.0, 65.0)),
                        })
                        step_s = 0.65 if (l_i == n_licks // 2 and n_licks > 12) else rng.uniform(0.11, 0.155)
                        lick_t += step_s
                    current_t = lick_t + rng.uniform(3.0, 5.0)
                else:
                    actual_hold = target_ms * rng.uniform(0.35, 0.85)
                    tone_on = np.nan
                    first_lick = np.nan
                    reward_ul = 0.0
                    outcome = "premature_exit"
                    current_t = v_enter + actual_hold / 1000.0 + rng.uniform(3.0, 5.0)

                trial_rows.append({
                    "trial_id": trial_counter,
                    "session_id": sess,
                    "mouse_id": mouse,
                    "trial_start_s": v_enter - 1.0,
                    "vertex_enter_s": v_enter,
                    "vertex_hold_ms": round(actual_hold, 2),
                    "hold_target_ms": target_ms,
                    "tone_onset_s": tone_on,
                    "tone_freq_hz": 8000.0,
                    "first_lick_s": first_lick,
                    "reward_delivered_ul": round(reward_ul, 2),
                    "outcome": outcome,
                    "opto_laser_on": opto_on,
                })
                trial_counter += 1

    trials_df = pd.DataFrame(trial_rows)
    licks_df = pd.DataFrame(lick_rows)

    total_duration_s = current_t + 5.0
    fs = 100.0
    t_neural = np.arange(0.0, total_duration_s, 1.0 / fs)
    bleach = 100.0 * np.exp(-t_neural / 3500.0)
    motion = 1.5 * np.sin(2.0 * np.pi * 0.25 * t_neural) + rng.normal(0, 0.3, len(t_neural))
    iso_405 = 0.85 * bleach + motion + rng.normal(0, 0.2, len(t_neural))

    neural_resp = np.zeros_like(t_neural)
    for ts in trials_df.loc[trials_df["outcome"] == "hit", "tone_onset_s"]:
        mask = (t_neural >= ts) & (t_neural <= ts + 2.5)
        neural_resp[mask] += 4.5 * np.exp(-(t_neural[mask] - ts) / 0.45)

    sig_465 = bleach + 1.15 * motion + neural_resp + rng.normal(0, 0.35, len(t_neural))
    photometry_df = pd.DataFrame({
        "timestamp_s": t_neural,
        "signal_465": sig_465,
        "isosbestic_405": iso_405,
    })

    fps = 30.0
    t_vid = np.arange(0.0, total_duration_s, 1.0 / fps)
    nose_x = 200.0 + 140.0 * np.sin(2.0 * np.pi * 0.08 * t_vid) + rng.normal(0, 2.0, len(t_vid))
    nose_y = 125.0 + 40.0 * np.cos(2.0 * np.pi * 0.08 * t_vid) + rng.normal(0, 2.0, len(t_vid))
    pose_df = pd.DataFrame({
        "frame_idx": np.arange(len(t_vid)),
        "timestamp_s": t_vid,
        "nose_x": nose_x,
        "nose_y": nose_y,
        "nose_likelihood": np.full(len(t_vid), 0.98),
        "midback_x": nose_x + 35.0,
        "midback_y": nose_y + 5.0,
        "midback_likelihood": np.full(len(t_vid), 0.99),
    })

    metadata = ExperimentMetadata(
        project_name="GPi_EPN_Striatum_3Input_Flexibility",
        circuit_target="GPi/EPN -> Dorsal Striatum",
        recording_modality="fiber_photometry",
        biosensor_or_opsin="jGCaMP7s + ChR2",
        sampling_rate_hz=fs,
        video_fps=fps,
        arena=ArenaGeometry(),
        sessions=phys_list,
        protocol_notes={
            "paradigm": "Arena rectangular con vértice triangular modulable -> tono -> licómetro opuesto (2-3 uL)",
            "schedule": "Sesiones cada 48 horas, mantenimiento alrededor de 80% peso corporal",
        },
        literature_notes=[
            "Smith et al. (2004) - Curva de aprendizaje espacio-estado EM e IO(0.95)",
            "Jean-Richard-dit-Bressel et al. (2020) / FiPhoPHA (2025) - Bootstrap + umbral consecutivo en PETH",
            "Musall et al. (2019) - Regresión Ridge en bloques (delta R^2) para separar movimiento de tarea",
        ],
    )
    return metadata, trials_df, licks_df, photometry_df, pose_df


if __name__ == "__main__":
    print("Generando datos sintéticos de demostración (3 inputs + fotometría + pose video)...")
    metadata, trials_df, licks_df, photometry_df, pose_df = generate_synthetic_dataset()
    results = run_complete_pipeline(
        task_name="vertex_cue_lick_3input",
        metadata=metadata,
        trials_df=trials_df,
        licks_df=licks_df,
        photometry_df=photometry_df,
        pose_df=pose_df,
        output_dir=Path("outputs_neuroflex"),
    )
    print("\n=== RESUMEN DE RESULTADOS ESTADÍSTICOS ===")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
