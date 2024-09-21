import random
from abc import ABCMeta, abstractmethod

from ..data_container import DataContainer


class TransformBase(metaclass=ABCMeta):
    def __init__(self, always_apply: bool, p: float) -> None:
        self.always_apply = always_apply
        self.p = p

    @abstractmethod
    def apply(self, data: DataContainer) -> DataContainer:
        raise NotImplementedError()

    def is_apply(self) -> bool:
        if self.always_apply or self.p > random.random():
            return True
        else:
            return False
