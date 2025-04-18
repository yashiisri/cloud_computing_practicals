# quantum_hello_world_statevector.py

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

def main():
    # Create a quantum circuit with 1 qubit and 1 classical bit
    qc = QuantumCircuit(1, 1)
    
    # Apply a Hadamard gate to create a superposition state
    qc.h(0)
    
    # Instead of using a simulator backend, compute the statevector directly
    state = Statevector.from_instruction(qc)
    
    # Get the probability distribution for each outcome from the statevector
    probs = state.probabilities_dict()
    print("Statevector probabilities:", probs)
    
    # Simulate measurement outcomes by sampling from the probability distribution
    shots = 1024
    outcomes = np.random.choice(list(probs.keys()), size=shots, p=list(probs.values()))
    
    # Count and display the number of times each outcome was observed
    unique, counts = np.unique(outcomes, return_counts=True)
    measurement_counts = dict(zip(unique, counts))
    print("Hello World!")
    print("Simulated measurement counts:", measurement_counts)

if __name__ == "__main__":
    main()
