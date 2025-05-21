import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ipywidgets import widgets
from gymnasium import Env, spaces
import gymnasium as gym
import math
import random



class FrozenLakeManipulationEnv(gym.Env):
    """Custom GridWorld environment with an agent and movable box."""
    
    def __init__(self, grid_size=4, agent_start=0, box_start=1, goal_agent=15, goal_box=15, holes=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.agent_start = agent_start
        self.box_start = box_start
        self.goal_agent = goal_agent
        self.goal_box = goal_box
        self.agent_pos = agent_start
        self.box_pos = box_start
        self.holding = False
        self.holes = holes if holes is not None else {5, 7, 11, 12}

        # Actions: up, down, left, right, grab, release
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.MultiDiscrete([self.n_states, self.n_states, 2])
    
    def reset(self):
        self.agent_pos = self.agent_start
        self.box_pos = self.box_start
        self.holding = False
        return self._get_obs(), {}

    def _get_obs(self):
        return (self.agent_pos, self.box_pos, int(self.holding))

    def _to_coord(self, pos):
        return divmod(pos, self.grid_size)

    def _to_index(self, row, col):
        return row * self.grid_size + col

    def _move(self, pos, action):
        row, col = self._to_coord(pos)
        if action == 0 and row > 0: row -= 1        # up
        elif action == 1 and row < self.grid_size - 1: row += 1  # down
        elif action == 2 and col > 0: col -= 1      # left
        elif action == 3 and col < self.grid_size - 1: col += 1  # right
        return self._to_index(row, col)

    def step(self, action):
        if action in [0, 1, 2, 3]:  # movement
            new_pos = self._move(self.agent_pos, action)
            if new_pos in self.holes or new_pos == self.agent_pos:
                return self._get_obs(), -1.0, True, False, {}
            self.agent_pos = new_pos
            if self.holding:
                self.box_pos = new_pos

        elif action == 4:  # grab
            if self.agent_pos == self.box_pos:
                self.holding = True

        elif action == 5:  # release
            self.holding = False

        # If agent or box is in a hole, terminate
        if self.agent_pos in self.holes or self.box_pos in self.holes:
            return self._get_obs(), -1.0, True, False, {}

        # Check goal condition
        success = (
            self.agent_pos == self.goal_agent and
            self.box_pos == self.goal_box and
            not self.holding
        )
        reward = 1.0 if success else 0.0
        return self._get_obs(), reward, success, False, {}
    
    