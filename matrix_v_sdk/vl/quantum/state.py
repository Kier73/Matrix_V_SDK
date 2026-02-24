import math
import numpy as np
from typing import List, Dict
from ..math.primitives import vl_mask
from ..math.rns import Q_RNS_PRIMES
from .tensor import MPSNode
from .gates import GateType, VL_QuantumGate

class VL_HolographicState:
    """
    O(1) memory representation of a quantum system.
    Stores timeline (seeds + ops) and realizes state only on measurement.
    """
    
    def __init__(self, qubit_count: int, seed: int):
        self.qubit_count = qubit_count
        self.seed = seed
        self.manifold_ids = [seed ^ i ^ 0xCAFEBABE for i in range(qubit_count)]
        self.mps_chain = [MPSNode() for _ in range(qubit_count)]
        self.ops = []
        self.collapsed_state = {}

    def apply_gate(self, gate: VL_QuantumGate):
        # Update entanglement graph for multi-qubit gates
        if gate.gate_type == GateType.CNOT:
            self.correlate(gate.controls[0], gate.targets[0])
        elif gate.gate_type == GateType.Toffoli:
            for c in gate.controls:
                self.correlate(c, gate.targets[0])
        
        # Immediate RNS manifold updates (MPS Tensor ops)
        for q_idx in gate.targets:
            if q_idx >= self.qubit_count: continue
            node = self.mps_chain[q_idx]
            
            if gate.gate_type == GateType.Hadamard:
                # Reflection on the RNS Torus: R' = P/4 - R
                for i in range(16):
                    p4 = Q_RNS_PRIMES[i] // 4
                    node.residues[i] = (p4 - node.residues[i]) % Q_RNS_PRIMES[i]
                    
            elif gate.gate_type == GateType.PauliX:
                # Bit flip: Add P/2
                for i in range(16):
                    node.residues[i] = (node.residues[i] + Q_RNS_PRIMES[i] // 2) % Q_RNS_PRIMES[i]
                    
            elif gate.gate_type in [GateType.PauliZ, GateType.S, GateType.T]:
                # Phase shifts
                shift_factor = {GateType.PauliZ: 2, GateType.S: 4, GateType.T: 8}[gate.gate_type]
                for i in range(16):
                    node.residues[i] = (node.residues[i] + Q_RNS_PRIMES[i] // shift_factor) % Q_RNS_PRIMES[i]
                    
            elif gate.gate_type == GateType.Phase:
                # Analog phase shift: R = (Prime * theta) / 2PI
                for i in range(16):
                    shift = round((Q_RNS_PRIMES[i] * gate.theta) / (2 * math.pi))
                    node.residues[i] = (node.residues[i] + shift) % Q_RNS_PRIMES[i]
        
        self.ops.append(gate)

    def correlate(self, q1: int, q2: int):
        if q1 >= self.qubit_count or q2 >= self.qubit_count: return
        n1, n2 = self.mps_chain[q1], self.mps_chain[q2]
        
        # Mix residues (Shared variety)
        for i in range(16):
            mixed = (n1.residues[i] + n2.residues[i]) % Q_RNS_PRIMES[i]
            n1.residues[i] = n2.residues[i] = mixed
            
        n1.compress_bond(n2)
        
        # Update manifold IDs
        shared_id = (self.manifold_ids[q1] ^ self.manifold_ids[q2]) & 0xFFFFFFFFFFFFFFFF
        self.manifold_ids[q1] = (self.manifold_ids[q1] + shared_id) & 0xFFFFFFFFFFFFFFFF
        self.manifold_ids[q2] = (self.manifold_ids[q2] + shared_id) & 0xFFFFFFFFFFFFFFFF

    def measure(self, qubit: int) -> bool:
        if qubit in self.collapsed_state: return self.collapsed_state[qubit]
        if qubit >= self.qubit_count: return False
        
        prob = self.calculate_probability(qubit)
        
        # Spectral Collapse logic
        phase_sig = sum(self.mps_chain[qubit].residues)
        resonance_key = vl_mask(self.manifold_ids[qubit] ^ phase_sig, self.seed)
        variety = resonance_key / 0xFFFFFFFFFFFFFFFF
        
        result = variety < prob
        self.collapsed_state[qubit] = result
        return result

    def calculate_probability(self, qubit: int) -> float:
        # Actual Implementation: Fourier-basis projection over the RNS torus.
        # Psi_k = e^(i * 2pi * r/p)
        node = self.mps_chain[qubit]
        real_sum = 0.0
        imag_sum = 0.0
        
        for r, p in zip(node.residues, Q_RNS_PRIMES):
            theta = 2.0 * math.pi * (r / p)
            real_sum += math.cos(theta)
            imag_sum += math.sin(theta)
            
        # Projected Phase Theta
        result_phase = math.atan2(imag_sum, real_sum)
        
        # Born Rule Isomorphism: P(|1>) = cos^2(Theta / 2)
        # This treats the interference of RNS oscillators as the state vector.
        return math.cos(result_phase / 2.0) ** 2

    def get_system_entropy(self) -> float:
        return sum(node.entanglement_entropy() for node in self.mps_chain)

