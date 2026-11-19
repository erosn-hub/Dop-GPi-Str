from __future__ import annotations

import json
import platform
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np
import pandas as pd


@dataclass
class ArenaGeometry:
    """Geometría de la arena rectangular con extremo triangular modulable."""
    length_cm: float = 40.0
    width_cm: float = 25.0
    triangle_vertex_xy_cm: Tuple[float, float] = (0.0, 12.5)
    vertex_tolerance_radius_cm: float = 3.0
    lickometer_xy_cm: Tuple[float, float] = (40.0, 12.5)
    pixels_per_cm: float = 10.0


@dataclass
class SessionPhysiology:
    """Control fisiológico por sesión (protocolo cada 2 días, ~80% peso corporal)."""
    session_id: str
    mouse_id: str
    session_date: str
    baseline_weight_g: float
    current_weight_g: float
    hours_since_last_session: float = 48.0
    supplemental_water_ml: float = 0.0
    group_condition: str = "control"

    @property
    def body_weight_pct(self) -> float:
        return 100.0 * (self.current_weight_g / self.baseline_weight_g)


@dataclass
class ExperimentMetadata:
    """Almacén persistente de parámetros, condiciones, circuito y notas bibliográficas."""
    project_name: str
    species: str = "Mus musculus"
    circuit_target: str = "GPi/EPN -> Striatum"
    recording_modality: str = "fiber_photometry"  # 'optogenetics' | 'calcium_imaging'
    biosensor_or_opsin: str = "jGCaMP7s"
    sampling_rate_hz: float = 100.0
    video_fps: float = 30.0
    reward_volume_ul_range: Tuple[float, float] = (2.0, 3.0)
    arena: ArenaGeometry = field(default_factory=ArenaGeometry)
    sessions: List[SessionPhysiology] = field(default_factory=list)
    protocol_notes: Dict[str, Any] = field(default_factory=dict)
    literature_notes: List[str] = field(default_factory=list)

    def to_json(self, output_path: Path) -> None:
        payload = asdict(self)
        for idx, s in enumerate(self.sessions):
            payload["sessions"][idx]["body_weight_pct"] = round(s.body_weight_pct, 3)
        payload["audit"] = {
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


TASK_REGISTRY: Dict[str, Type["BehaviorTaskProtocol"]] = {}


def register_task(name: str) -> Callable[[Type["BehaviorTaskProtocol"]], Type["BehaviorTaskProtocol"]]:
    """Decorador para registrar nuevas pruebas conductuales sin alterar el código base."""
    def decorator(cls: Type["BehaviorTaskProtocol"]) -> Type["BehaviorTaskProtocol"]:
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


class BehaviorTaskProtocol(ABC):
    """Clase base abstracta para tareas conductuales extensibles."""

    def __init__(self, metadata: ExperimentMetadata) -> None:
        self.metadata = metadata

    @abstractmethod
    def validate_and_preprocess(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Valida columnas requeridas y calcula variables derivadas ensayo por ensayo."""

    @abstractmethod
    def alignment_events(self) -> Dict[str, str]:
        """Devuelve mapeo {etiqueta_evento: columna_timestamp_en_trials_df}."""

    @abstractmethod
    def primary_stimulus_column(self) -> str:
        """Columna principal modulada (ej. hold_target_ms o tone_freq_hz)."""


@register_task("vertex_cue_lick_3input")
class VertexCueLickTask(BehaviorTaskProtocol):
    """
    Protocolo de 3 acciones/inputs:
    1) Aproximación al vértice triangular y permanencia modulable (X ms).
    2) Emisión de tono/frecuencia auditiva.
    3) Respuesta en licómetro opuesto (2-3 uL) con conteo y tiempos de licks.
    """

    REQUIRED_COLUMNS = {
        "trial_id", "session_id", "mouse_id", "trial_start_s",
        "vertex_enter_s", "vertex_hold_ms", "hold_target_ms",
        "tone_onset_s", "tone_freq_hz", "first_lick_s",
        "reward_delivered_ul", "outcome",
    }

    def validate_and_preprocess(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(trials_df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en trials_df: {sorted(missing)}")

        df = trials_df.copy().sort_values(["mouse_id", "session_id", "trial_start_s"]).reset_index(drop=True)
        norm_outcome = df["outcome"].astype(str).str.lower()
        df["is_hit"] = (norm_outcome == "hit").astype(int)
        df["is_premature"] = (norm_outcome == "premature_exit").astype(int)
        df["hold_error_ms"] = df["vertex_hold_ms"] - df["hold_target_ms"]
        df["reaction_transit_ms"] = (df["first_lick_s"] - df["tone_onset_s"]) * 1000.0
        df.loc[df["is_hit"] == 0, "reaction_transit_ms"] = np.nan

        if "opto_laser_on" not in df.columns:
            df["opto_laser_on"] = 0
        if "session_num" not in df.columns:
            df["session_num"] = df.groupby("mouse_id")["session_id"].transform(lambda s: pd.factorize(s)[0] + 1)

        if self.metadata.sessions:
            weight_map = {(s.mouse_id, s.session_id): s.body_weight_pct for s in self.metadata.sessions}
            group_map = {(s.mouse_id, s.session_id): s.group_condition for s in self.metadata.sessions}
            keys = list(zip(df["mouse_id"], df["session_id"]))
            df["body_weight_pct"] = [weight_map.get(k, np.nan) for k in keys]
            if "group_condition" not in df.columns:
                df["group_condition"] = [group_map.get(k, "default") for k in keys]
        return df

    def alignment_events(self) -> Dict[str, str]:
        return {
            "Entrada_Vertice": "vertex_enter_s",
            "Tono_Auditivo": "tone_onset_s",
            "Primer_Lick_Recompensa": "first_lick_s",
        }

    def primary_stimulus_column(self) -> str:
        return "hold_target_ms"
