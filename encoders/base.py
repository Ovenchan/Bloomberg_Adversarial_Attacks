from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, images):
        """Encode a batch of images into feature representations."""
        raise NotImplementedError