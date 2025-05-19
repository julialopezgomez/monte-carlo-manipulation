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
from environments import FrozenLakeManipulationEnv, GripperDiscretisedEnv
from data_loading import to_one_hot_encoding


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
    def __init__(self, n_states, n_actions, hidden_dim=1024):
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
    
    
    
class LearnedMCTSNode:
    def __init__(self, 
                 state,
                 make_env,
                 net,
                 parent=None, 
                 action=None, 
                 prior=0.0,
                 cpuct=1.41,
                 device='cpu',
                 verbose=False):
        
        self.state = state
        self.parent = parent
        self.action = action            # action taken to reach this node
        self.prior = prior              # prior probability of this action
        self.children = {}
        
        self.N = defaultdict(int)       # visit counts per action
        self.W = defaultdict(float)     # total reward per action
        self.Q = defaultdict(float)     # average reward per action (Q = W/N)
        self.reward = 0.0
        self.terminal = False
        
        self.puct_constant = cpuct
        
        self.make_env = make_env
        self.env = make_env()
        self.net = net
        self.device = device
        self.verbose = verbose

    def is_leaf(self):
        return len(self.children) == 0

    def puct(self):
        """Calculate the PUCT value for this node.

        Returns:
            puct_value (float): The PUCT value for this node.
        """
        
        if self.parent.N[self.action] == 0:
            return float("inf")
        
        total_N = sum(self.parent.N.values())
        
        exploitation = self.parent.Q[self.action]
        exploration = self.puct_constant * self.prior * math.sqrt(total_N) / (1 + self.parent.N[self.action])
        
        return exploitation + exploration
        

    def best_puct_child(self):
        return max(self.children.values(), key=lambda child: child.puct())

    def best_child(self):
        return max(self.children.values(), key=lambda child: self.Q[child.action])

    def selection(self):
        """Traverse the tree to select a promising node to expand.
        
        Returns:
            - node  (MCTSNode): The selected node to expand.
            - is_goal (bool): True if the node is a goal state.
        """
        node = self
        while not node.is_leaf():
            node = node.best_puct_child()
            if self.verbose:
                print(f"Selected node {node.state} with visits {node.parent.N[node.action]} and value {node.parent.Q[node.action]}")
        
        return node
    
    
    def expand(self):
        """Expand the current node by simulating the environment and adding child nodes.

        Returns:
            child (LearnedMCTSNode): The child node that is a goal state, if any.
        """
        
        # s_tensor = F.one_hot(torch.tensor(self.state), self.env.observation_space.n).float().to(self.device)
        s_tensor = to_one_hot_encoding(self.state, self.env.observation_space).float().to(self.device)
            
        with torch.no_grad():
            logp, _ = self.net(s_tensor.unsqueeze(0))
            p = torch.exp(logp).cpu().numpy()[0]
        
        for action in range(self.env.action_space.n):
            env_copy = self.make_env()
            env_copy.reset()
            env_copy.unwrapped.s = self.state
            
            obs, reward, terminated, truncated, _ = env_copy.step(action)

            # Only add child if it is not a (non successful) terminal state or if it is not the same state
            # if reward == 0 and terminated or self.state == obs:
            #     continue

            child = LearnedMCTSNode(obs,  
                                    make_env=self.make_env,
                                    parent=self,
                                    action=action, 
                                    prior=p[action],
                                    cpuct=self.puct_constant,
                                    device=self.device,
                                    net=self.net,
                                    verbose=self.verbose)
            
            self.children[action] = child

            if terminated or obs == self.state:
                child.terminal = True
                child.reward = reward if reward == 1 else -1
                
            if reward == 1:
                return child
            
        if self.verbose:
            print(f"Expanded node {self.state} with children: {[c.state for c in self.children.values()]}")
        
        return None
    

    def evaluation(self):
        """Evaluate the current node using the neural network.
        Returns:
            value (float): The value of the current node.
        """
        # s_tensor = F.one_hot(torch.tensor(self.state), self.env.observation_space.n).float().to(self.device)
        s_tensor = to_one_hot_encoding(self.state, self.env.observation_space).float().to(self.device)
        with torch.no_grad():
            _, value = self.net(s_tensor.unsqueeze(0))
            
        return value.item() 

    def backpropagation(self, value):
        """Propagate the simulation result back up the tree."""
        node = self
        while node:
            parent = node.parent
            if parent:
                parent.N[node.action] += 1
                parent.W[node.action] += value
                parent.Q[node.action] = parent.W[node.action] / parent.N[node.action]
                node = parent
            else:
                break 