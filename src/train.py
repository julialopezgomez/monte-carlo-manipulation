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
from mcts_models import LearnedMCTSNode#, MCTSDataset
from torch.utils.data import DataLoader, Dataset


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


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
    states, pis, zs = batch['state'], batch['policy'], batch['value']
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
          buffer_size=3000,
          eval_interval=10,
          num_eval=50,
          num_sims=100,
          target_sr=0.95,
          cpuct=1.41,
          tau=1.,
          device='cpu',
          verbose=False):
    """Train the network using self-play and MCTS."""
    all_losses = []
    best_sr = -1.0  # Initialize best win rate
    env = make_env()

    # Initialize dataset and dataloader
    replay_dataset = MCTSDataset(max_size=buffer_size)

    # Use tqdm for the main episode loop
    with tqdm(range(1, num_episodes + 1), desc="Training Episodes", unit="episode") as episode_pbar:
        for episode in episode_pbar:
            # ---- 1) Self-play: Add new data to dataset ----
            data, z = self_play_episode(net, make_env, device)
            for s, pi in data:
                replay_dataset.append(s, pi, z)  # Appends to dataset

            train_loader = DataLoader(
                replay_dataset,
                batch_size=batch_size,
                shuffle=True,  # Automatically shuffles each epoch
                num_workers=2, # Set to 0 for simpler debugging, increase for performance
            )
            # ---- 2) Training: Use DataLoader for batches ----
            # tqdm for the training batches within an episode
            with tqdm(train_loader, desc=f"  Episode {episode} Training", unit="batch", leave=False) as train_pbar:
                for batch in train_loader:
                    # Unpack batch

                    # Forward pass and training
                    p_loss, v_loss = train_on_batch(batch, net, env, opt, device=device)
                    all_losses.append((p_loss, v_loss))

                    train_pbar.set_postfix(p_loss=f"{p_loss:.4f}", v_loss=f"{v_loss:.4f}")
                    train_pbar.update() # Manually update as we are inside another tqdm

            # ---- 3) Evaluation
            # ---- PERIODIC EVALUATION ----
            if episode % eval_interval == 0:
                wins = 0
                # tqdm for the evaluation games
                with tqdm(range(num_eval), desc=f"  Episode {episode} Evaluation", unit="game", leave=False) as eval_pbar:
                    for i in range(num_eval):
                        env = make_env()
                        state, _ = env.reset()
                        done = False
                        while not done:
                            pi = run_mcts(
                                state, net,
                                make_env=make_env,
                                num_sims=num_sims, cpuct=cpuct, tau=tau,
                                device=device, verbose=verbose,
                            )
                            a = int(np.argmax(pi))
                            state, reward, term, trunc, _ = env.step(a)
                            done = term or trunc

                        if reward == 1: # Assuming reward of 1 means a win
                            wins += 1

                        current_sr = wins / (i + 1)
                        eval_pbar.set_postfix(wins=f"{wins}/{i+1}", sr=f"{current_sr:.2f}")
                        eval_pbar.update() # Manually update

                sr = wins / num_eval
                episode_pbar.write(f"Ep {episode}: eval_sr={sr:.2f} (best={best_sr:.2f})") # Write evaluation result below the main bar

                # checkpoint & report back to the episode bar
                if sr > best_sr:
                    best_sr = sr
                    # torch.save(net.state_dict(), f"models/checkpoint_ep{episode}.pt") # Uncomment to save checkpoints
                    # print(f"New best model saved at episode {episode} with SR: {best_sr:.2f}") # Optional print

                if sr >= target_sr:
                    episode_pbar.write(f"Target SR reached at ep {episode} ({sr:.2f})")
                    break # Exit the main episode loop

            # Update the main episode progress bar description with current status
            episode_pbar.set_postfix(latest_p_loss=f"{all_losses[-1][0]:.4f}" if all_losses else "N/A",
                                     latest_v_loss=f"{all_losses[-1][1]:.4f}" if all_losses else "N/A",
                                     buffer_size=len(replay_dataset),
                                     best_sr=f"{best_sr:.2f}")

    print("\nTraining finished.")
    # You might return all_losses or other metrics here
    return all_losses, best_sr