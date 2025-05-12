import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
from tqdm import tqdm
from mcts_models import LearnedMCTSNode, MCTSDataset
from torch.utils.data import DataLoader, Dataset

def run_mcts(root_state,
             net,
             make_env,
             num_sims=1000,
             cpuct=1.41,
             tau=1.,
             device='cpu',
             verbose=False):

    root = LearnedMCTSNode(
                    state=root_state,
                    make_env=make_env,
                    cpuct=cpuct,
                    device=device,
                    verbose=verbose
                    )

    # expand root with network evaluation:
    value = root.expand(net)
    if verbose:
        print(f"Root node {root.state} expanded with value {value}")

    for _ in range(num_sims):
        node = root

        # Selection
        while not node.is_leaf():
            prev_node = node
            action, node = node.select()
            if verbose:
                print(f"Selected action {action} for node {prev_node.state} -> {node.state}")

        # Expansion and evaluation with network
        value = node.expand(net)

        if verbose:
            print(f"Node {node.state} expanded with value {value}")

        # Backpropagation (up one level)
        assert node.action == action, f"Expected action {action}, but got {node.action}"
        node.parent.backpropagate(node.action, value)

        if verbose:
            print(f"Finished backpropagation of value {value} for action {action} in node {node.state}")

    # build visit‐count distribution π
    counts = np.array([root.N[a] for a in range(root.nA)], dtype=np.float32)
    # apply temperature
    counts = counts**(1/tau)
    pi = counts / counts.sum()

    if verbose:
        print(f"Visit counts: {root.N}")
        print(f"Action probabilities: {pi}")

    return pi

def self_play_episode(net, make_env, device='cpu', verbose=False):
    """Run one game of self-play, return list of training tuples (s,π), and z."""
    data = []
    env = make_env()
    nA = env.action_space.n
    state, _ = env.reset()
    done = False
    i = 0

    while not done:
        if verbose:
            print(f"Step {i}: state {state}")

        # get action probabilities from MCTS
        if verbose:
            print(f"Running MCTS for state {state}")
        pi = run_mcts(state,
                        net,
                        make_env=make_env,
                        num_sims=100,
                        cpuct=1.41,
                        tau=1.0,
                        device=device,
                        verbose=verbose
                      )

        if verbose:
            print(f"Action probabilities: {pi}")

        # store (s,π) pair
        data.append((state, pi))

        # pick action (you can sample or argmax)
        action = np.random.choice(nA, p=pi)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if verbose:
            print(f"Action {action} taken, new state {state}, reward {reward}")

            if done:
                print(f"Game ended with reward {reward}")

        # update state
        state = obs

    # z = final reward (0 for fail, 1 for success)
    z = float(reward)
    return data, z


def train_on_batch(batch, net, env, opt, device='cpu'):
    states, pis, zs = zip(*batch)
    nS = env.observation_space.n

    # one-hot encode states
    X = F.one_hot(torch.tensor(states), nS).float()

    target_pi = torch.tensor(pis, dtype=torch.float32, device=device)
    target_z  = torch.tensor(zs, dtype=torch.float32, device=device)

    opt.zero_grad()
    logp, v = net(X)
    loss_p = (-target_pi * logp).sum(dim=1).mean()
    loss_v = F.mse_loss(v, target_z)
    loss   = loss_p + loss_v
    loss.backward()
    opt.step()
    return loss_p.item(), loss_v.item()


def train(net,
          make_env,
          opt,
          num_episodes=1000,
          batch_size=64,
          buffer_size=10000,
          num_sims=100,
          cpuct=1.41,
          tau=1.,
          device='cpu',
          verbose=False):
    """Train the network using self-play and MCTS."""
    env = make_env()
    all_losses = []
    
    # Initialize dataset and dataloader
    replay_dataset = MCTSDataset(max_size=buffer_size)
    train_loader = DataLoader(
        replay_dataset,
        batch_size=batch_size,
        shuffle=True,  # Automatically shuffles each epoch
        num_workers=2  # Parallel data loading (optional)
    )

    for episode in range(1, num_episodes + 1):
        # ---- 1) Self-play: Add new data to dataset ----
        data, z = self_play_episode(net, make_env, device)
        for s, pi in data:
            replay_dataset.append(s, pi, z)  # Appends to dataset

        # ---- 2) Training: Use DataLoader for batches ----
        for batch in tqdm(train_loader, desc=f"Training (Ep {episode})"):
            states, policies, values = batch
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            
            # Forward pass and training (replace with your logic)
            p_loss, v_loss = train_on_batch(states, policies, values, net, opt)
            all_losses.append((p_loss, v_loss))


