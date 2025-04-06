import numpy as np
from typing import Tuple, Dict, List
import numpy as np

class MarkovChains:
    def __init__(self, states: List[str]):
        self.states = sorted(states)
        self.n_states = len(states)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.init_probs = None # pi_j
        self.transition_matrix = None # A_jk

    def fit(self, tokenized_corpus: List[List[str]]):
        self.initial_counts = np.zeros(self.n_states, dtype=np.float32) # N_1j
        for tokens in tokenized_corpus:
            if tokens[0] in self.state_to_idx:
                self.initial_counts[self.state_to_idx[tokens[0]]] += 1

        self.init_probs = self.initial_counts / self.initial_counts.sum() # pi_j

        self.transition_counts = np.zeros((self.n_states, self.n_states), dtype=np.float32) # N_jk
        self.transition_matrix = np.zeros((self.n_states, self.n_states), dtype=np.float32) # A_jk

        for sequence in tokenized_corpus:
            for t in range(len(sequence)-1):
                current_state = sequence[t]
                next_state = sequence[t+1]
                
                if current_state in self.state_to_idx and next_state in self.state_to_idx:
                    i = self.state_to_idx[current_state]
                    j = self.state_to_idx[next_state]
                    self.transition_counts[i, j] += 1

        row_sums = self.transition_counts.sum(axis=1)
        for i in range(self.n_states):
            if row_sums[i] > 0:
                self.transition_matrix[i, :] = self.transition_counts[i, :] / row_sums[i]
            else:
                # If a state has no observed transitions, we assign uniform probabilities
                self.transition_matrix[i, :] = 1. / self.n_states


    def predict_next_state(self, current_state: str) -> Tuple[str, np.array]:
        if current_state not in self.state_to_idx:
            raise ValueError(f"Unknown state: {current_state}")
        i = self.state_to_idx[current_state]
        next_state_probs = self.transition_matrix[i, :]
        next_state_idx = next_state_probs.argmax()
        return self.states[next_state_idx], next_state_probs

    def generate(self, start: str=None, length: int=10) -> List[str]:
        if self.init_probs is None or self.transition_matrix is None:
            raise ValueError("The model is not fitted yet.")

        if start is None:
            start = np.random.choice(self.states, p=self.init_probs)

        sequence=[start]
        for _ in range(length-1):
            current_state = sequence[-1]
            next_state, _ = self.predict_next_state(current_state=current_state)
            sequence.append(next_state)
        
        return sequence

    def score_sequence(self, sequence: List[str]) -> float:
        assert len(sequence)>0, "the sequence is empty"
        score = self.init_probs[self.state_to_idx[sequence[0]]]

        for t in range(1, len(sequence)):
            current_state = sequence[t-1]
            next_state = sequence[t]
            
            if current_state not in self.state_to_idx or next_state not in self.state_to_idx:
                # raise ValueError(f"Unknown states: {current_state} and/or {next_state}")
                return 0.
            i = self.state_to_idx[current_state]
            j = self.state_to_idx[next_state]
            score *= self.transition_matrix[i, j]

        return score
    
    def perplexity(self, sequence: List[str]) -> float:
        n = len(sequence)
        return self.score_sequence(sequence=sequence)**(-1//n)
                
