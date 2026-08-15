import numpy as np
from dataclasses import dataclass



@dataclass
class Image:
    """Base class for all types of images."""



@dataclass
class CompressedImage(Image):
    """Compressed image."""



@dataclass
class Frame:
    """One single camera frame."""

    timestamp: float | None = None
    """The Unix timestamp that this frame is captured."""
    time: str | None = None
    """The human-readable time in form of dd-mm-yyyy:hh-mm-ss."""
    frame: np.ndarray | None = None
    """The RGB frame data, shaoe (H, W, 3)."""
