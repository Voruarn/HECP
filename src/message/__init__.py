from dataclasses import dataclass
from typing import Union


@dataclass
class BaseTrainerMessage:
    max_epochs: int
    num_nodes: int
    num_devices: int


@dataclass
class BaseDataModuleMessage:
    max_steps: int


@dataclass
class SupervisionDataModuleMessage(BaseDataModuleMessage):
    classes: Union[dict, list]