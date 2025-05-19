import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import deque, defaultdict, namedtuple
import torch
from gymnasium.spaces import Discrete, MultiDiscrete

def to_one_hot_encoding(state, obs_space):
    """
    state: a single observation, either
      - an int (for Discrete), or
      - a sequence of ints (for MultiDiscrete)
    obs_space: gym.spaces.Discrete or MultiDiscrete
    returns: FloatTensor of shape [input_dim]
    """
    if isinstance(obs_space, Discrete):
        # single discrete var → one-hot
        idx = torch.tensor(state, dtype=torch.long)
        return F.one_hot(idx, num_classes=int(obs_space.n)).float()

    if isinstance(obs_space, MultiDiscrete):
        arr = list(state)
        nvec = obs_space.nvec.tolist()
        parts = []
        for i, (val, n) in enumerate(zip(arr, nvec)):
            if n == 2:
                # final dimension: binary 0/1 pass-through
                parts.append(torch.tensor([val], dtype=torch.float32))
            else:
            # discrete var → one-hot
                idx = torch.tensor(val, dtype=torch.long)
                parts.append(F.one_hot(idx, num_classes=n).float())
                
        return torch.cat(parts, dim=0)

    raise TypeError(f"Unsupported space: {type(obs_space)}")


class ReplayDataset(Dataset):
    """Wrap a list of (state, policy, outcome) tuples,
       automatically one-hot-encoding states via obs_space."""
       
    def __init__(self, experiences, obs_space):
        """
        experiences: list of namedtuples with .state, .mcts_policy, .game_outcome
        obs_space:    gym.spaces.Discrete or MultiDiscrete matching state shape
        """
        self.exps      = experiences
        self.obs_space = obs_space

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        state, policy, outcome = self.exps[idx]
        x   = to_one_hot_encoding(state, self.obs_space)
        pi = torch.tensor(policy,  dtype=torch.float32)
        z = torch.tensor(outcome, dtype=torch.float32)
        return x, pi, z



class ReplayBuffer:
    ''' The ReplayBuffer stores experience tuples for neural network training.
        Each experience tuple contains:
            - The state (observation) from the environment at a specific step.
            - The MCTS-derived policy (visit counts distribution) for that state.
            - The final outcome (value) of the game episode this state belongs to.
    '''
    def __init__(self, buffer_size, sample_size):
        """
        Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        self.memory = deque(maxlen=buffer_size)
        self.sample_size = sample_size
        # Define the structure of an experience tuple to store state, policy, and value
        self.experience = namedtuple("Experience", field_names=["state", "mcts_policy", "game_outcome"])


    # Correct the add method signature and assignment
    def add(self, state, mcts_policy, value):
        """Add a new experience (state, mcts_policy, value) to memory."""
        # Create an experience tuple using the defined structure and correct arguments
        e = self.experience(state, mcts_policy, value)
        self.memory.append(e) # Add the tuple to the buffer

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        n = min(len(self.memory), self.sample_size)
        return random.sample(self.memory, k=n)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)