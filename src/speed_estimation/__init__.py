"""Vehicle speed estimation with perspective correction (Isogun & Agagu)."""
from .config import SOURCE, TARGET, PipelineConfig
from .evaluation import evaluate
from .pipeline import SpeedPipeline, process_video, summarize_tracks
from .speed import SpeedEstimator, window_speed_kmh
from .transform import NaiveScaleTransformer, ViewTransformer

__all__ = [
    "SOURCE", "TARGET", "PipelineConfig", "evaluate", "SpeedPipeline", "process_video",
    "summarize_tracks", "SpeedEstimator", "window_speed_kmh", "NaiveScaleTransformer",
    "ViewTransformer",
]
__version__ = "1.0.0"
