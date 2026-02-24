from enum import Enum, auto
from typing import List, Union, Optional

class GateType(Enum):
    Hadamard = auto()
    PauliX = auto()
    PauliY = auto()
    PauliZ = auto()
    S = auto()
    T = auto()
    CNOT = auto()
    Phase = auto()
    Toffoli = auto()
    Measure = auto()

class VL_QuantumGate:
    def __init__(self, gate_type: GateType, targets: List[int], theta: Optional[float] = None, controls: List[int] = None):
        self.gate_type = gate_type
        self.targets = targets
        self.theta = theta
        self.controls = controls or []

    def __repr__(self):
        return f"{self.gate_type.name}({self.targets}, theta={self.theta}, controls={self.controls})"

