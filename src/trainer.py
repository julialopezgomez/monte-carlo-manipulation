import os
import datetime
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import deque, defaultdict
from tqdm import tqdm
from train import AlphaLoss, train_pipeline
from environments import FrozenLakeManipulationEnv, GripperDiscretisedEnv
from data_loading import to_one_hot_encoding, ReplayBuffer, ReplayDataset
from mcts_models import MCTSNode, LearnedMCTSNode, AlphaZeroNet
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training script")
    
    
    # MCTS / AlphaZero params
    parser.add_argument("--num-sims",    type=int,   default=5000,
                        help="MCTS simulations per self-play step")
    parser.add_argument("--num-self-play", type=int, default=8,
                        help="Self-play games to generate per epoch")
    parser.add_argument("--num-epochs",  type=int,   default=20,
                        help="Number of training epochs")
    parser.add_argument("--cpuct",       type=float, default=1.41,
                        help="PUCT exploration constant")
    parser.add_argument("--tau",         type=float, default=1.0,
                        help="Temperature for π = N^(1/τ)")
    # Training params
    parser.add_argument("--batch-size",  type=int,   default=128,
                        help="Batch size for optimizer")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Eval every N self-play games")
    parser.add_argument("--target-sr",   type=float, default=0.90,
                        help="Target success rate to stop training")
    parser.add_argument("--regularization", type=float, default=1e-4,
                        help="L2 weight decay")
    parser.add_argument("--max-episodes", type=int, default=10,
                        help="Maximum episodes per training run")
    parser.add_argument("--num-eval",    type=int,   default=50,
                        help="Number of games for evaluation")
    parser.add_argument("--buffer-size", type=int,   default=20000,
                        help="Replay buffer capacity")
    parser.add_argument("--sample-size", type=int,   default=1024,
                        help="Minibatch sample size")

    # Environment choice
    parser.add_argument("--env", choices=["FL", "FLM", "GM"],
                        default="FL",
                        help="Which environment to create")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()



def main():
    args = parse_args()

    # Unpack for convenience (optional)
    NUM_SIMS       = args.num_sims
    NUM_SELF_PLAY  = args.num_self_play
    NUM_EPOCHS     = args.num_epochs
    CPUCT          = args.cpuct
    TAU            = args.tau

    BATCH_SIZE     = args.batch_size
    LR             = args.lr
    EVAL_INTERVAL  = args.eval_interval
    TARGET_SR      = args.target_sr
    REGULARIZATION = args.regularization
    MAX_EPISODES   = args.max_episodes

    NUM_EVAL       = args.num_eval
    BUFFER_SIZE    = args.buffer_size
    SAMPLE_SIZE    = args.sample_size

    env_type       = args.env
    VERBOSE        = args.verbose
    # create environment based on choice
    if env_type == "FL":
        def make_env():
            return gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    elif env_type == "FLM":
        def make_env():
            return FrozenLakeManipulationEnv()
    elif env_type == "GM":
        def make_env():
            return GripperDiscretisedEnv()
    else:
        raise ValueError(f"Unknown env: {env_type}")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    env    = make_env()
    env.reset()
    state, _ = env.reset()
    nA = env.action_space.n

    if isinstance(env.observation_space, Discrete):
        nS = env.observation_space.n
    else:
        # Assuming the observation space is a tuple of (states, ..., states, holding/not_holding) 
        print(f"len(env.observation_space.sample()): {len(env.observation_space.sample())}")
        print(f"int((len(env.observation_space.sample()) - 1) * env.n_states): {(len(env.observation_space.sample()) - 1) * env.n_states}")
        nS = int((len(env.observation_space.sample()) - 1) * env.n_states) + 1
    net    = AlphaZeroNet(nS, nA).to(device)
    optimizer   = optim.Adam(net.parameters(), lr=LR, weight_decay=REGULARIZATION)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


    
    root_node = LearnedMCTSNode(state=state,
                            make_env=make_env,
                            net=net,
                            cpuct=CPUCT,
                            device=device,
                            verbose=VERBOSE)

    
    train_pipeline(net=net,
                make_env=make_env,
                optimizer=optimizer,
                scheduler=scheduler,
                buffer_size=BUFFER_SIZE,
                sample_size=SAMPLE_SIZE,
                batch_size=BATCH_SIZE,
                num_sims=NUM_SIMS,
                num_epochs=NUM_EPOCHS,
                tau=TAU,
                cpuct=CPUCT,
                num_episodes=MAX_EPISODES,
                num_self_play=NUM_SELF_PLAY,
                eval_interval=EVAL_INTERVAL,
                num_eval=NUM_EVAL,
                target_sr=TARGET_SR,
                device=device)

if __name__ == "__main__":
    main()