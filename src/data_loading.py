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
        return F.one_hot(idx, num_classes=obs_space.n).float()

    if isinstance(obs_space, MultiDiscrete):
        arr = list(state)
        nvec = obs_space.nvec.tolist()
        parts = []
        for i, (val, n) in enumerate(zip(arr, nvec)):
            if i == len(nvec) - 1:
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


'''The ReplayBuffer stores experience tuples for neural network training.
Each experience tuple contains:
    - The state (observation) from the environment at a specific step.
    - The MCTS-derived policy (visit counts distribution) for that state.
    - The final outcome (value) of the game episode this state belongs to.
'''
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples (state, mcts_policy, game_outcome)."""

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # Define the structure of an experience tuple to store state, policy, and value
        self.experience = namedtuple("Experience", field_names=["state", "mcts_policy", "game_outcome"])


    # Correct the add method signature and assignment
    def add(self, state, mcts_policy, game_outcome):
        """Add a new experience (state, mcts_policy, game_outcome) to memory."""
        # Create an experience tuple using the defined structure and correct arguments
        e = self.experience(state, mcts_policy, game_outcome)
        self.memory.append(e) # Add the tuple to the buffer

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            # Cannot sample a batch if there aren't enough experiences
            # You might want to handle this case, e.g., return empty list or raise error
            # For now, let's return a sample of whatever is available if less than batch_size
             if len(self.memory) == 0:
                 return []
             experiences = random.sample(self.memory, k=len(self.memory))
        else:
             experiences = random.sample(self.memory, k=self.batch_size)

        # When sampling for training, you typically want to return tensors
        # Assuming state, policy, and outcome are numpy arrays or similar that can be converted to tensors
        # You might need to adjust dtype based on your specific data types
        states = torch.stack([torch.tensor(e.state, dtype=torch.float32) for e in experiences])
        policies = torch.stack([torch.tensor(e.mcts_policy, dtype=torch.float32) for e in experiences])
        outcomes = torch.stack([torch.tensor(e.game_outcome, dtype=torch.float32) for e in experiences])

        # Return the batch as separate tensors, which is convenient for training
        return states, policies, outcomes


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __getitem__(self, idx):
         """Allows accessing individual experiences by index."""
         # This method is typically needed if you use torch.utils.data.DataLoader directly with the buffer
         # However, your sample method already returns a batch ready for training.
         # If you use DataLoader, you might remove the sample method and rely on DataLoader's batching.
         # For now, let's add it to make it Dataset-like, returning tensors.
         experience = self.memory[idx]
         state = torch.tensor(experience.state, dtype=torch.float32)
         policy = torch.tensor(experience.mcts_policy, dtype=torch.float32)
         outcome = torch.tensor(experience.game_outcome, dtype=torch.float32)
         return state, policy, outcome