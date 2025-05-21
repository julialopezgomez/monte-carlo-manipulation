import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import math
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
from tqdm import tqdm
from mcts_models import LearnedMCTSNode
from data_loading import ReplayBuffer, ReplayDataset, to_one_hot_encoding
from torch.utils.data import DataLoader, Dataset


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        # value_error = (value - y_value) ** 2
        # policy_error = torch.sum((-policy* 
        #                         (1e-6 + y_policy.float()).float().log()), 1)
        # total_error = value_error + policy_error
        
        
        # Value loss (MSE)
        value_loss = F.mse_loss(y_value, value)

        # Policy loss (cross-entropy from log-prob to soft target)
        policy_loss = -(policy * y_policy).sum(dim=1).mean()

        # Total loss (weighted sum, equal weights here)
        total_error = value_loss + policy_loss
        return total_error


def run_mcts(root_node,
             tau=1.0, 
             num_sims=1000,
             pipeline_verbose=False):
    """Run MCTS simulations from the given node."""
    
    root_node.expand()
    
    for _ in range(num_sims):
        node = root_node.selection()

        if pipeline_verbose:
            print(f"\nSELECTED NODE: {node.state}, with visits {node.parent.N[node.action]} and value {node.parent.Q[node.action]}\n")
        
        if node.terminal:
            if pipeline_verbose: 
                print(f"Terminal node reached: {node.parent.state} -> {node.state}, with reward {node.reward}")
            node.backpropagation(node.reward)
            continue
        
        # Check if the node had been visited before
        if node.parent.N[node.action] > 0:
            # If the node has been visited before, expand it
            goal_node = node.expand()
            
            if pipeline_verbose:
                print(f"\nEXPANDED NODE: {node.state}, with children {[c.state for c in node.children.values()]}")
        
            # If the node is a goal state, select it, otherwise select a random child
            node = goal_node if goal_node is not None else node.best_puct_child()
            if pipeline_verbose:
                print(f"selected node {node.state} from children {[c.state for c in node.parent.children.values()]}.")
                print(f"is goal node: {goal_node is not None}\n")
            
            
        # If the node is a terminal state, use its reward as the value
        if node.terminal:
            value = node.reward
            if pipeline_verbose:
                print(f"BACKPROPAGATING REWARD: {value} from terminal node {node.state}")
        else:
            value = node.evaluation() # get value from NN
            if pipeline_verbose:
                print(f"BACKPROPAGATING VALUE: {value} from non-terminal node {node.state}")
            
        node.backpropagation(value)
        
        
        
    counts = np.array([root_node.N[a] for a in range(root_node.env.action_space.n)])
    
    counts = counts**(1 / tau)
    
    pi = counts / counts.sum()
    
    return pi

def self_play_episode(
    make_env,
    net,
    num_sims=100,
    tau=1.,
    cpuct=1.41,
    device='cpu',
    verbose=False
):
    data = []
    env = make_env()
    state, _ = env.reset()
    done = False
    
    while not done:
        root_node = LearnedMCTSNode(state=state,
                                    make_env=make_env,
                                    net=net,
                                    cpuct=cpuct,
                                    device=device,
                                    verbose=verbose)
        
        pi = run_mcts(root_node, tau=tau, num_sims=num_sims)
        
        action = np.random.choice(np.arange(len(pi)), p=pi)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        data.append((state, pi))
        
        if terminated or truncated:
            done = True
            
        state = next_state
        
    return data, reward

def train(net, dataloader, device,
          optimizer,
          scheduler,
          epoch_start=0, epoch_stop=20, cpu=0):
    """
    Train the AlphaZero network using MCTS-generated dataset.

    Args:
        net: Neural network model.
        dataset: Training dataset (raw data to be wrapped with board_data).
        device: torch.device (e.g., 'cuda' or 'cpu').
        optimizer: torch.optim optimizer (e.g., Adam).
        scheduler: torch.optim.lr_scheduler instance.
        epoch_start: Starting epoch index.
        epoch_stop: Stopping epoch index.
        cpu: Random seed / CPU identifier.
    """
    # Set random seed for reproducibility
    torch.manual_seed(cpu)
    net.train()

    # Use custom loss function
    criterion = AlphaLoss()

    losses_per_epoch = []

    # Outer progress bar for epochs
    epoch_bar = tqdm(range(epoch_start, epoch_stop), desc="Epochs", position=0)
    for epoch in epoch_bar:

        total_loss = 0.0
        losses_per_batch = []

        # Inner progress bar for batches
        batch_bar = tqdm(enumerate(dataloader, 0),
                         total=len(dataloader),
                         desc=f"Epoch {epoch + 1}",
                         leave=False,
                         position=1)

        for i, data in batch_bar:
            state, policy, value = data

            # Move tensors to GPU or CPU
            state = state.to(device).float()
            policy = policy.to(device).float()
            value = value.to(device).float()

            # Forward + backward + optimization step
            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred, value, policy_pred, policy)
            loss.backward()
            optimizer.step() 
            scheduler.step()  # Step the learning rate scheduler

            # Track total loss for this batch
            total_loss += loss.item()
            batch_bar.set_postfix(loss=loss.item())

            # Periodic logging every 10 batches
            if i % 10 == 9:
                avg_loss = total_loss / 10
                losses_per_batch.append(avg_loss)
                total_loss = 0.0

        # Epoch-level loss tracking
        if losses_per_batch:
            epoch_avg_loss = sum(losses_per_batch) / len(losses_per_batch)
            losses_per_epoch.append(epoch_avg_loss)
            epoch_bar.set_postfix(avg_epoch_loss=epoch_avg_loss)

        # Early stopping criterion (very conservative)
        if len(losses_per_epoch) > 100:
            recent = sum(losses_per_epoch[-4:-1]) / 3
            earlier = sum(losses_per_epoch[-16:-13]) / 3
            if abs(recent - earlier) <= 0.01:
                tqdm.write("Early stopping criterion met.")
                break

    # Final loss vs. epoch plot
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1, len(losses_per_epoch) + 1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    os.makedirs("model_data/", exist_ok=True)
    plt.savefig(os.path.join("model_data/", f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"))
    
    
def evaluate(net, make_env, num_episodes=50, device='cpu'): 
    """Evaluate the trained model on the environment.

    Args:
        net: Neural network model.
        make_env: Function to create a new environment instance.
        num_episodes: Number of episodes to evaluate.

    Returns:
        success_rate: Success rate of the model in the environment.
    """
    
    success_count = 0
    net.eval()
    
    with tqdm(total=num_episodes, desc="Evaluating", position=0) as pbar:
        for episode in range(num_episodes):
            env = make_env()
            state, _ = env.reset()
            
            node = LearnedMCTSNode(state=state,
                                    make_env=make_env,
                                    net=net,
                                    device=device)

            done = False
            
            while not done:
                # Run MCTS to get the policy
                logp, _ = net(to_one_hot_encoding(state, env.observation_space).float().to(device).unsqueeze(0))
                p = torch.exp(logp).cpu().detach().numpy()[0]
                action = np.argmax(p)
                next_state, reward, terminated, truncated, _ = env.step(action)
                node = LearnedMCTSNode(state=next_state,
                                        make_env=make_env,
                                        net=net,
                                        parent=node,
                                        action=action,
                                        prior = p[action],
                                        device=device)
                
                done = terminated or truncated
                state = next_state
            # Check if the episode was successful
            if reward == 1:
                success_count += 1
            
            pbar.set_postfix(success_rate=success_count / (episode + 1))
            pbar.update(1)
            
            
    success_rate = success_count / num_episodes
    
    return success_rate


def train_pipeline( net,
                    make_env,
                    optimizer,
                    scheduler,
                    buffer_size=20000,
                    sample_size=2048,
                    batch_size=128,
                    num_sims=100,
                    num_epochs=20,
                    tau=1.0,
                    cpuct=1.41,
                    num_episodes=10,
                    num_self_play=10,
                    eval_interval=1,
                    num_eval=200,
                    target_sr=0.90,
                    device = 'cpu',
                    verbose = False):
    
    env = make_env()
    best_net = copy.deepcopy(net)
    
    replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                 sample_size=sample_size)
    
    for episode in range(num_episodes):
        
        # ------------- Self-play -------------
        self_play_bar = tqdm(range(num_self_play),
                            desc=f"Episode {episode+1}/{num_episodes}",
                            position=0)

        for g in self_play_bar:
            data, reward = self_play_episode(make_env=make_env,
                                            net=net,
                                            num_sims=num_sims,
                                            tau=tau,
                                            cpuct=cpuct,
                                            device=device,
                                            verbose=verbose)

            for state, pi in data:
                # Convert state to one-hot encoding
                replay_buffer.add(state=state,
                                mcts_policy=pi,
                                value=reward)

            if verbose:
                tqdm.write(f"Episode {episode+1}. Self-play episode {g+1} finished with reward {reward}. Buffer size: {len(replay_buffer)}")
            
            self_play_bar.set_postfix(reward=reward, buffer=len(replay_buffer))
            
            
        # ------------- Training -------------
        if len(replay_buffer) < batch_size:
            continue
        
        net.train()
        
        experiences = replay_buffer.sample()
        dataset = ReplayDataset(experiences, obs_space=env.observation_space)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train the network
        train(net=net,
              dataloader=dataloader,
              device=device,
              optimizer=optimizer,
              scheduler=scheduler,
              epoch_start=0, 
              epoch_stop=num_epochs,
              cpu=0)
        
        # ------------- Evaluation -------------
        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate(net, make_env, num_episodes=num_eval, device=device)
            print(f"Episode {episode+1}. Success rate: {success_rate:.2f}")
            
            # Save the best model and stop
            if success_rate >= target_sr:
                best_net = copy.deepcopy(net)
                torch.save(best_net.state_dict(), os.path.join("models/", f"best_model_{episode+1}.pth"))
                print(f"Best model saved at episode {episode+1}.")
                break