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
from collections import deque, defaultdict


# --- MCTS Node Class ---
class MCTSNode:
    def __init__(self, state, parent=None, action=None, make_env=None, verbose=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.make_env = make_env
        self.verbose = verbose

    def is_leaf(self):
        return len(self.children) == 0

    def uct(self, exploration=1.41):
        if self.visits == 0 or self.parent is None:
            return float("inf")
        exploitation = self.value / self.visits
        exploration_bonus = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_bonus

    def best_uct_child(self):
        return max(self.children, key=lambda child: child.uct())

    def best_child(self):
        return max(self.children, key=lambda child: child.value)

    def selection(self, env):
        """Traverse the tree to select a promising node to expand."""
        node = self
        while not node.is_leaf():
            node = node.best_uct_child()
        if node.visits == 0:
            return node, None
        goal_node = node.expand(env)
        return (goal_node if goal_node else random.choice(node.children), goal_node is not None)

    def expand(self, env):
        """Expand the node by trying all possible actions."""
        for action in range(env.action_space.n):
            env_copy = self.make_env()
            env_copy.reset()
            env_copy.unwrapped.s = self.state
            obs, reward, terminated, truncated, _ = env_copy.step(action)

            if reward == 0 and terminated or self.state == obs:
                continue

            child = MCTSNode(obs, parent=self, action=action, make_env=self.make_env, verbose=self.verbose)
            self.children.append(child)

            if reward == 1:
                if self.verbose:
                    print(f"Goal found from state {self.state} with action {action} → {obs}")
                return child

        if self.verbose:
            print(f"Expanded node {self.state} with children: {[c.state for c in self.children]}")
        return None

    def simulation(self, max_steps=10):
        """Perform a rollout from the current node using random actions."""
        env_copy = self.make_env()
        env_copy.reset()
        env_copy.unwrapped.s = self.state
        obs = self.state

        for _ in range(max_steps):
            action = env_copy.action_space.sample()
            obs, reward, terminated, truncated, _ = env_copy.step(action)

            if self.verbose:
                print(f"Simulating from {self.state} → {obs} with action {action} reward {reward}")

            if reward == 1 or terminated or truncated:
                return reward
        return 0

    def backpropagation(self, reward):
        """Propagate the simulation result back up the tree."""
        node = self
        while node:
            node.visits += 1
            node.value += reward
            if self.verbose:
                print(f"Backprop node {node.state}, visits={node.visits}, value={node.value}")
            node = node.parent
            
            
# --- MCTS Search Class ---
class VanillaMCTS:
    def __init__(self, make_env, num_iterations=100, num_simulations=10, exploration=1.41, verbose=False):
        self.make_env = make_env
        self.env = make_env()
        self.root = MCTSNode(self.env.reset()[0], make_env=self.make_env, verbose=verbose)
        self.num_iterations = num_iterations
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.verbose = verbose
        self.root.expand(self.env)

    def run(self):
        for _ in range(self.num_iterations):
            node, goal = self.root.selection(self.env)
            
            # If the node is a goal node, backpropagate with reward 1
            if goal:
                node.backpropagation(1)
                if self.verbose:
                    print(f"Goal reached at state {node.state}")
                break
            
            # If the node is not a goal, perform simulations and backpropagate the result
            reward = node.simulation(max_steps=self.num_simulations)
            node.backpropagation(reward)
            
        if self.verbose:
            print(f"Finished {self.num_iterations} iterations.")
            

class AlphaZeroNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # policy head
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        # value head
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: one-hot or feature vector of shape (batch, n_states)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        p = F.log_softmax(self.policy_head(h), dim=1)  # log-probs
        v = torch.tanh(self.value_head(h))             # in [-1,1]
        return p, v.squeeze(-1)
    
# --- MCTS Node Class ---
class LearnedMCTSNode:
    def __init__(self,
                 state,
                 make_env,
                 parent=None,
                 action=None,
                 prior=0.0,
                 cpuct=1.41,
                 device='cpu',
                 verbose=False):

        self.state = state
        self.make_env = make_env
        self.env = make_env()
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.parent = parent
        self.action = action         # action taken to reach this node
        self.prior   = prior          # P(s,a) from network
        self.cpuct  = cpuct          # exploration constant
        self.children= {}             # action → child_node
        self.N       = defaultdict(int)   # visit counts per child
        self.W       = defaultdict(float) # total value per child
        self.Q       = defaultdict(float) # mean value per child
        self.device = device
        self.verbose = verbose

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, net):
        # get network outputs for this state
        s_tensor = F.one_hot(torch.tensor([self.state], device=self.device), self.nS).float()
        s_tensor = s_tensor.to(self.device)
        logp, value = net(s_tensor)
        p = logp.exp().detach().cpu().numpy()[0]

        # create children with priors
        for action in range(self.nA):
            if action not in self.children:


                # Copy the environment by creating a new instance
                env_copy = self.make_env()
                env_copy.reset()
                env_copy.unwrapped.s = self.state

                # Perform the action in the copied environment
                obs, reward, terminated, truncated, _ = env_copy.step(action)

                # if np.array_equal(obs, self.state):
                #     # If the state is the same, we don't need to create a new child
                #     continue

                self.children[action] = LearnedMCTSNode(
                    state=obs,
                    make_env=self.make_env,
                    parent=self,
                    action=action,
                    prior=p[action],
                    cpuct=self.cpuct,
                    device=self.device,
                    verbose=self.verbose
                    )

                if self.verbose:
                    print(f"Expanding node {self.state} with action {action} and prior {p[action]}")

        if self.verbose:
            print(f"Expanded node {self.state} with children: {[c.state for c in self.children.values()]}")
            print(f"Node {self.state} has value {value.item()} and prior {p}")
        return value.item()

    def select(self):
        # pick action that maximizes Q + U
        total_N = sum(self.N.values())
        best_action, best_score = None, -float('inf')
        for action, child in self.children.items():
            U = self.c_puct * child.prior * math.sqrt(total_N) / (1 + self.N[action])
            score = self.Q[action] + U
            if score > best_score:
                best_score, best_action = score, action

        if self.verbose:
            print(f"Selected action {best_action} with score {best_score} (Q={self.Q[best_action]}, U={U})")

        return best_action, self.children[best_action]

    def backpropagate(self, action, value):
        """ Backpropagate the value of the simulation to this node.

        Args:
            action (int): The action taken to reach this node.
            value (float): The network-predicted value for child node.
        """
        self.W[action] += value
        self.N[action] += 1
        self.Q[action] = self.W[action] / self.N[action]

        if self.verbose:
            print(f"Backpropagating value {value} for action {action} in node {self.state}")
            print(f"Node {self.state} updated: W={self.W[action]}, N={self.N[action]}, Q={self.Q[action]}")

        node = self
        if node.parent:
            node.parent.backpropagate(self.action, value)
            
            
            
class MCTSDataset(Dataset):
    def __init__(self, max_size=None):
        self.data = []  # Stores (state, policy, value) tuples
        self.max_size = max_size  # Optional: Limit dataset size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, pi, z = self.data[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32)
        )

    def append(self, state, policy, value):
        if self.max_size and len(self.data) >= self.max_size:
            # remove the oldest sample which z is not > 0
            i = 0
            while i < len(self.data) and self.data[i][2] > 0:
                i += 1
            if i < len(self.data):
                self.data.pop(i)
            else:
                self.data.pop(0)
        self.data.append((state, policy, value))