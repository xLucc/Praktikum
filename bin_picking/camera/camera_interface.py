from abc import ABC, abstractmethod
import numpy as np

class Camera(ABC):

    @abstractmethod
    def get_rgb(self)->list:
        pass

    @abstractmethod
    def get_depth(self)-> list:
        pass

