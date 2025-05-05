
#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
import argparse
import os
from tqdm import trange, tqdm
import wandb



class PolicyValueMCTS(nn.Module):
    """Neural network for policy and value prediction"""
    def __init__(self, state_dim, action_dim):
        super(PolicyValueMCTS, self).__init__()
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Policy head
        self.policy_head = nn.Linear(128, action_dim)
        
        # Value head
        self.value_head = nn.Linear(128, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Shared features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy output (log probabilities)
        policy = F.log_softmax(self.policy_head(x), dim=-1)
        
        # Value output (scalar)
        value = torch.tanh(self.value_head(x))
        
        return policy, value



class MCTSNode:
    """Node for Monte Carlo Tree Search with neural guidance"""
    def __init__(self, state, parent=None, action=None, prior=0, env=None, 
                 policy_net=None, value_net=None, device='cpu'):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior  # Prior probability from policy network
        self.children = []
        self.visits = 0
        self.total_value = 0.0  # Cumulative value from simulations
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = device
        self.expanded = False
    
    def is_leaf(self):
        return not self.expanded
    
    def uct_score(self, exploration_weight=1.41):
        """Calculate UCT score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        # Use value network prediction as heuristic
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
            _, value = self.value_net(state_tensor)
            heuristic = value.item()
        
        exploitation = self.total_value / self.visits
        exploration = exploration_weight * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        return exploitation + exploration + 0.1 * heuristic  # Combine with value net prediction
    
    def select_child(self):
        """Select child with highest UCT score"""
        return max(self.children, key=lambda child: child.uct_score())
    
    def expand(self):
        """Expand the node using policy network predictions"""
        if self.expanded:
            return
            
        # Get action probabilities from policy network
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy_net(state_tensor)
            action_probs = torch.exp(action_probs).cpu().numpy().flatten()
        
        # Create child nodes for all possible actions
        for action in range(self.env.action_space.n):
            # Create a copy of the environment to simulate the action
            env_copy = copy.deepcopy(self.env)
            env_copy.reset()
            env_copy.unwrapped.s = self.state
            
            # Take the action
            next_state, reward, terminated, truncated, _ = env_copy.step(action)
            
            # Skip invalid states (where state doesn't change)
            if np.array_equal(next_state, self.state):
                continue
                
            # Create child node with prior from policy network
            child = MCTSNode(
                next_state, 
                parent=self, 
                action=action, 
                prior=action_probs[action],
                env=self.env,
                policy_net=self.policy_net,
                value_net=self.value_net,
                device=self.device
            )
            self.children.append(child)
        
        self.expanded = True
    
    def rollout(self, max_depth=10000):
        """Perform a rollout (simulation) from this node"""
        env_copy = copy.deepcopy(self.env)
        env_copy.reset()
        env_copy.unwrapped.s = self.state
        
        total_reward = 0
        depth = 0
        
        while depth < max_depth:
            # Use policy network for rollout actions (with some randomness)
            state_tensor = torch.FloatTensor(env_copy.unwrapped.s).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, _ = self.policy_net(state_tensor)
                action_probs = torch.exp(action_probs).cpu().numpy().flatten()
            
            # Add some noise for exploration
            action = np.random.choice(len(action_probs), p=action_probs)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward
            depth += 1
            
            if total_reward == 1 or terminated or truncated:
                return total_reward
                
            # # Update state
            # env_copy.unwrapped.s = next_state
        
        return total_reward
    
    def backpropagate(self, value):
        """Backpropagate the simulation result up the tree"""
        self.visits += 1
        self.total_value += value
        
        if self.parent:
            self.parent.backpropagate(value)
            
            
            
class NeuralMCTS:
    """Neural Monte Carlo Tree Search algorithm"""
    def __init__(self, env, policy_net, value_net, device='cpu', 
                 num_simulations=1000, exploration_weight=1.0, 
                 rollout_depth=10000, lr=0.001):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = device
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        
        # Optimizers
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = deque(maxlen=10000)
    
    def search(self, state):
        """Perform MCTS from given state"""
        root = MCTSNode(
            state, 
            env=self.env,
            policy_net=self.policy_net,
            value_net=self.value_net,
            device=self.device
        )
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection
            while not node.is_leaf():
                node = node.select_child()
            
            # Expansion
            if node.visits > 0:  # Only expand nodes that have been visited before
                node.expand()
                
                # If we expanded into a terminal state, use the actual reward
                if len(node.children) == 0:  # Terminal state
                    value = 0  # Terminal states get 0 value (we already got the reward)
                else:
                    # Perform a rollout from the expanded node
                    value = node.rollout(self.rollout_depth)
            else:
                # For unvisited nodes, use value network prediction
                state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value = self.value_net(state_tensor)
                    value = value.item()
                
                # Also do a rollout to get more accurate value estimate
                rollout_value = node.rollout(self.rollout_depth)
                value = 0.5 * value + 0.5 * rollout_value  # Combine estimates
            
            # Backpropagation
            node.backpropagate(value)
            
            # Store experience for training
            if node.parent is not None:  # Not the root node
                self.buffer.append((
                    node.parent.state,
                    node.action,
                    value,
                    node.state
                ))
        
        return root
    
    def update_networks(self, batch_size=32):
        """Update policy and value networks using experience"""
        if len(self.buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.buffer, batch_size)
        states, actions, values, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        log_probs, _ = self.policy_net(states)
        policy_loss = -torch.mean(log_probs.gather(1, actions.unsqueeze(1)) * values.unsqueeze(1))
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        _, pred_values = self.value_net(states)
        value_loss = F.mse_loss(pred_values.squeeze(), values)
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def get_action(self, state, temperature=1.0):
        """Get action from MCTS search"""
        root = self.search(state)
        
        # Get visit counts for each action
        visit_counts = np.zeros(self.env.action_space.n)
        for child in root.children:
            visit_counts[child.action] = child.visits
        
        # Apply temperature to visit counts
        if temperature == 0:
            action = np.argmax(visit_counts)
        else:
            visit_probs = visit_counts ** (1/temperature)
            visit_probs /= visit_probs.sum()
            action = np.random.choice(len(visit_probs), p=visit_probs)
        
        return action
    
    
class ManipulationEnv(gym.Env):
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
        return np.array([self.agent_pos, self.box_pos, int(self.holding)])

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



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int,   default=50,  help="training episodes")
    p.add_argument("--max-steps", type=int, default=20,   help="max steps per episode")
    p.add_argument("--budget",    type=int, default=200,  help="MCTS rollout budget")
    p.add_argument("--update-fq", type=int, default=5,    help="network update frequency")
    p.add_argument("--use-wandb", action="store_true",    help="log to wandb")
    p.add_argument("--verbose",   action="store_true",    help="print debug info")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) init wandb
    if args.use_wandb:
        wandb.init(project="manipulation_mcts", config=vars(args))

    # 2) device, env, mcts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    env = ManipulationEnv()                             
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n


    policy_net = PolicyValueMCTS(state_dim, action_dim).to(device)
    value_net = PolicyValueMCTS(state_dim, action_dim).to(device)
    
    # Create MCTS
    mcts = NeuralMCTS(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        device=device,
        num_simulations=5000,
        exploration_weight=1.0,
        rollout_depth=10000,
        lr=0.001
    )

    # 3) storage
    all_rewards, all_p_losses, all_v_losses = [], [], []

    # 4) outer loop: episodes
    for ep in trange(args.episodes, desc="Episodes"):
        state, _ = env.reset()
        ep_reward = 0

        # wrap inner loop in tqdm if verbose
        steps = trange(args.max_steps, desc="Steps", leave=False) if args.verbose else range(args.max_steps)
        for step in steps:
            action = mcts.get_action(state, temperature=1.0)
            next_state, reward, done, trunc, _ = env.step(action)
            ep_reward += reward

            # push experience
            mcts.buffer.append((state, action, reward, next_state))
            state = next_state

            # every k steps update nets
            if step % args.update_fq == 0:
                p_loss, v_loss = mcts.update_networks()
                if p_loss is not None:
                    all_p_losses.append(p_loss)
                    all_v_losses.append(v_loss)
                    if args.use_wandb:
                        wandb.log({"policy_loss": p_loss, "value_loss": v_loss, "step": ep*args.max_steps+step})

            if done or trunc:
                break

        all_rewards.append(ep_reward)
        if args.use_wandb:
            wandb.log({"episode_reward": ep_reward, "episode": ep})

        # occasionally test greedy policy
        if ep % 10 == 0:
            avg_test = test_policy(env, mcts)
            print(f"[TEST] ep{ep:03d}  trainR={ep_reward:.2f}  testR={avg_test:.2f}")
            if args.use_wandb:
                wandb.log({"test_reward": avg_test, "episode": ep})

    # 5) save final policy/value nets
    os.makedirs("models", exist_ok=True)
    torch.save(mcts.policy_value_net.state_dict(), "models/policy_value_net.pt")

    # 6) plot or flush wandb
    if args.use_wandb:
        wandb.finish()

    print("Training done. Rewards:", np.mean(all_rewards[-10:]))

def test_policy(env, mcts, num_episodes=10):
    tot = 0
    for _ in range(num_episodes):
        s, _ = env.reset()
        done=False
        while not done:
            a = mcts.get_action(s, temperature=0.0)  # greedy
            s, r, done, trunc, _ = env.step(a)
            tot += r
    return tot/num_episodes

if __name__=="__main__":
    main()
