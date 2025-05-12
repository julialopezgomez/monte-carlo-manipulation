import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class FrozenLakeManipulationEnv(gym.Env):
    """Custom GridWorld environment with an agent and movable box."""
    
    def __init__(self, grid_size=4, agent_start=0, box_start=1, goal_agent=13, goal_box=15, holes=None):
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
            if new_pos in self.holes:
                return self._get_obs(), 0.0, True, False, {}
            if self.holding:
                self.agent_pos = new_pos
                self.box_pos = new_pos
            else:
                self.agent_pos = new_pos

        elif action == 4:  # grab
            if self.agent_pos == self.box_pos:
                self.holding = True

        elif action == 5:  # release
            self.holding = False

        # If agent or box is in a hole, terminate
        if self.agent_pos in self.holes or self.box_pos in self.holes:
            return self._get_obs(), 0.0, True, False, {}

        # Check goal condition
        success = (
            self.agent_pos == self.goal_agent and
            self.box_pos == self.goal_box and
            not self.holding
        )
        reward = 1.0 if success else 0.0
        return self._get_obs(), reward, success, False, {}
    
    
class GripperDiscretisedEnv(gym.Env):
    """Discretized environment for gripper rotating and controlling a cap."""

    def __init__(self, angle_steps=18, finger_steps=7):
        super().__init__()

        # Discretization
        self.angle_steps = angle_steps  # number of discrete angles (e.g., 18 for 20-degree steps)
        self.finger_steps = finger_steps  # number of discrete gripper openings (0=-0.055, 6=-0.025)

        # Define angle and gripper ranges
        self.gripper_min_width = -0.055
        self.gripper_max_width = -0.025

        # Define the action space: rotate+/-, open/close gripper
        self.action_space = spaces.Discrete(4)

        # Define the observation space: (cap_angle_idx, gripper_angle_idx, gripper_width_idx)
        self.observation_space = spaces.MultiDiscrete(
            [self.angle_steps, self.angle_steps, self.finger_steps]
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cap_angle_idx = 0
        self.gripper_angle_idx = 0
        self.gripper_width_idx = 6  # fully open
        return self._get_obs(), {}

    def _get_obs(self):
        return (self.cap_angle_idx, self.gripper_angle_idx, self.gripper_width_idx)

    def _idx_to_angle(self, idx):
        return idx * (int(360 / self.angle_steps))

    def _idx_to_width(self, idx):
        return self.gripper_min_width + idx * (self.gripper_max_width - self.gripper_min_width) / (self.finger_steps - 1)

    def is_grasping(self):
        return self.gripper_width_idx == 6  # grasping if nearly closed

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if action == 0:  # Rotate +20 deg
            self.gripper_angle_idx = (self.gripper_angle_idx + 1) % self.angle_steps
            if self.is_grasping():
                self.cap_angle_idx = (self.cap_angle_idx + 1) % self.angle_steps

        elif action == 1:  # Rotate -20 deg
            self.gripper_angle_idx = (self.gripper_angle_idx - 1) % self.angle_steps
            if self.is_grasping():
                self.cap_angle_idx = (self.cap_angle_idx - 1) % self.angle_steps

        elif action == 2:  # Open gripper
            if self.gripper_width_idx > 0:
                self.gripper_width_idx -= 1

        elif action == 3:  # Close gripper
            if self.gripper_width_idx < self.finger_steps - 1:
                self.gripper_width_idx += 1

        # Check if goal is achieved
        success = (self.cap_angle_idx == 3) and (self.gripper_width_idx == 4)
        reward = 1.0 if success else 0.0
        
        # terminate if a collision is detected
        def convert_angle_to_radians(angle):
            return (angle * np.pi) / 180
        
        collision = visualizer.check_collision_q_by_ik([
            convert_angle_to_radians(self._idx_to_angle(self.cap_angle_idx)),
            convert_angle_to_radians(self._idx_to_angle(self.gripper_angle_idx)),
            self._idx_to_width(self.gripper_width_idx),
        ])
        
        if collision:
            return self._get_obs(), 0.0, False, True, {}

        return self._get_obs(), reward, success, False, {}