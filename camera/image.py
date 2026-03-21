from dataclasses import dataclass


@dataclass
class Image:
    """Base class for all types of images."""



@dataclass
class CompressedImage(Image):
    """Compressed image."""