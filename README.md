# Dec-POMDP MARL with Resource Competition

## Overview
This project implements a **multi-agent reinforcement learning (MARL) framework** for a **decentralized partially observable Markov decision process (Dec-POMDP)**. The environment models **resource competition** where villagers collect food while thieves attempt to steal from them. The framework integrates **PettingZoo, Gymnasium, RLlib, and Pygame** to support **multi-agent policy learning, real-time simulation, and visualization**.

## Features
- **Dec-POMDP Environment**: A **grid-based competitive survival scenario** where agents have **partial observability**.
- **Multi-Agent RL Training**: Uses **Proximal Policy Optimization (PPO)** via **Ray's RLlib**, enabling **separate policies** for villagers and thieves.
- **Dynamic Resource Allocation**: Villagers **gather and trade**, while thieves **steal and evade**, leading to emergent adversarial behavior.
- **Real-Time Visualization**: A **Pygame-based renderer** for tracking agent interactions.
- **Modular & Extendable**: Easily integrates with new agent roles, policies, and interaction mechanics.

## Technologies Used
- **Reinforcement Learning**: **RLlib (PPO)** for multi-agent policy training.
- **Multi-Agent Simulation**: **PettingZoo** for Dec-POMDP agent interactions.
- **Environment Modeling**: **Gymnasium** for defining grid-world dynamics.
- **Visualization**: **Pygame** for real-time rendering.
- **Parallel Execution**: **Ray Tune** for scalable RL training.

## Installation
```bash
# Clone the repository
git clone https://github.com/arda-kara/Dec_POMDP-MARL-with-Resource-Competition.git
cd Dec_POMDP-MARL-with-Resource-Competition

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Train MARL Policies
```bash
python train_marl.py
```
### Run Simulation & Visualization
```bash
python main.py
python pygame_renderer.py  # Optional: Visualize agent behavior
```

## Repository Structure
- `envs/`: **Dec-POMDP environment** definitions.
- `train_marl.py`: **PPO-based RL training** for villagers & thieves.
- `pygame_renderer.py`: **Real-time visualization** of grid-based agent interactions.
- `rllib_env_wrapper.py`: **RLlib wrapper** for PettingZoo compatibility.
- `main.py`: **Simulation runner** with customizable configurations.

## Research & Applications
This project is designed for **multi-agent learning, strategic decision-making, and adversarial AI**. It is applicable to **game theory, decentralized control, and AI-based resource management**.

## License
This project is licensed under the **MIT License**.

