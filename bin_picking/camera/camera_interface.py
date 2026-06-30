from abc import ABC, abstractmethod
import numpy as np


class Camera(ABC):
    """Abstract base class for all camera backends used in the bin-picking pipeline."""

    @abstractmethod
    def get_color(self) -> list:
        """Capture and return one or more BGR color frames as a list of numpy arrays."""
        pass

    @abstractmethod
    def get_depth(self) -> list:
        """Capture and return one or more raw uint16 depth frames as a list of numpy arrays."""
        pass

