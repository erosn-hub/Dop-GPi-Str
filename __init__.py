"""
neuroflex_stats: Paquete estadístico modular para evaluación conductual,
neural (fotometría / imagenología de calcio / optogenética) y videográfica.
"""

from .core import (
    ArenaGeometry,
    BehaviorTaskProtocol,
    ExperimentMetadata,
    SessionPhysiology,
    TASK_REGISTRY,
    VertexCueLickTask,
    register_task,
)
from .behavior_stats import (
    HypothesisTestSelector,
    LickMicrostructureAnalyzer,
    LickMicrostructureConfig,
    PsychometricChronometricAnalyzer,
    StateSpaceLearningEstimator,
)
from .neural_video import (
    PhotometryPETHAnalyzer,
    VideoKinematicEncodingAnalyzer,
)
from .viz_and_pipeline import (
    ExactNeuroPlotter,
    run_complete_pipeline,
)

__all__ = [
    "ArenaGeometry",
    "BehaviorTaskProtocol",
    "ExperimentMetadata",
    "SessionPhysiology",
    "TASK_REGISTRY",
    "VertexCueLickTask",
    "register_task",
    "HypothesisTestSelector",
    "LickMicrostructureAnalyzer",
    "LickMicrostructureConfig",
    "PsychometricChronometricAnalyzer",
    "StateSpaceLearningEstimator",
    "PhotometryPETHAnalyzer",
    "VideoKinematicEncodingAnalyzer",
    "ExactNeuroPlotter",
    "run_complete_pipeline",
]
