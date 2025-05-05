# Monte-Carlo Tree Search for Robotic Manipulation

This repository contains an implementation of Monte-Carlo Tree Search (MCTS) applied to representative robotic manipulation tasks. It includes both discrete grid-world proofs-of-concept and a geometric gripper manipulation environment.

---

## Features

* **Neural-guided MCTS**
  Combines tree search with learned policy/value networks and rollouts.

* **Manipulation Environments**

  * `gripper_env/`: discrete gripper pick-and-place environment
  * Custom 4×4 grid world with movable box
  * FrozenLake baseline implementation

* **Training & Evaluation Scripts**

  * `manipulation_frozen_lake_neural_script.py`: standalone training loop with `tqdm` progress bars and WandB logging
  * CLI flags for episodes, rollout budget, update frequency, verbosity

* **Interactive Notebooks**

  * `frozen_lake.ipynb`: baseline grid-world experiments
  * `manipulation_planner_3dof.ipynb`, `manipulation_planner_4dof.ipynb`: planning in 3-DOF and 4-DOF manipulators
  * `manipulation_planner_TD_3dof.ipynb`: temporal-difference control variant

* **Visualization Utilities**

  * `ciris_plant_visualizer.py`: render and inspect manipulation scenes
  * `visualization_utils.py`: shared plotting functions

* **Containerized Environment**

  * `Dockerfile` for reproducible setup with GPU support

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/mcts-manipulation.git
   cd mcts-manipulation
   ```

2. **Docker (optional if not using gripper env)**

   ```bash
   docker build -t mcts_manipulation .  
   docker run --gpus all -it --rm mcts_manipulation bash
   ```

3. **Python environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt ### Needs creating
   ```

---

## Usage

### Training with Script

```bash
python manipulation_frozen_lake_neural_script.py \
  --episodes 100 \
  --max-steps 50 \
  --budget 1000 \
  --update-fq 10 \
  --use-wandb \
  --verbose
```

* `--episodes`, `--max-steps`, `--budget`: control MCTS rollout count and episode length
* `--update-fq`: network update frequency
* `--use-wandb`: enable Weights & Biases logging
* `--verbose`: enable inner-loop progress bars

### Interactive Notebooks

Open any `.ipynb` file in Jupyter to explore variants:

* **FrozenLake**: `frozen_lake.ipynb`
* **Manipulation FrozenLake**: in `manipulation_frozen_lake.ipynb`, `manipulation_frozen_lake_hierarchical.ipynb`, `manipulation_frozen_lake_manipulation_neural.ipynb`
* **3-DOF & 4-DOF MIQP Planners**: `manipulation_planner_3dof.ipynb`, `manipulation_planner_4dof.ipynb`
* **TD Control Baseline**: `manipulation_planner_TD_3dof.ipynb`
* **Learning MCTS**: `manipulation_frozen_lake_neural.ipynb`

---

## Main Files

* **`manipulation_frozen_lake_neural_script.py`**
  Standalone training script combining MCTS (`NeuralMCTS`), policy/value networks, and CLI-driven configuration.

* **`manipulation_frozen_lake.ipynb`**

* **`frozen_lake.ipynb`**
  Proof-of-concept MCTS on the OpenAI Gym FrozenLake environment.

* **`manipulation_planner_3dof.ipynb` / `manipulation_planner_4dof.ipynb`**
  MCTS planners for 3-DOF and 4-DOF manipulation gripper scene demo (not MCTS).

* **`manipulation_planner_TD_3dof.ipynb`**
  Temporal-difference learning baseline for the 3-DOF manipulation task.

* **`gripper_env/`**
  Custom discrete environment where an agent controls a gripper to pick and place an object.

* **`ciris_plant_visualizer.py`**
  Utility to render robotic scenes and trajectories used in the gripper environment.

* **`visualization_utils.py`**
  Common plotting routines for value functions, reward curves, and search trees.

* **`Dockerfile`**
  Defines a container with GPU-enabled PyTorch and all dependencies for reproducible experiments.

* **`requirements.txt`** #TODO
  Lists Python packages required for training, visualization, and logging.

