# Connect_X_AI

This project implements intelligent reinforcement learning agents to play [Kaggle ConnectX](https://www.kaggle.com/competitions/connectx), a strategic grid-based game inspired by Connect Four. The goal is to align a specified number of checkers in a row—vertically, horizontally, or diagonally—on a variable-sized board.

## ✨ Features

- Trains agents using two main RL methods:
  - **Double Deep Q-Network (Double DQN)**
  - **Proximal Policy Optimization (PPO)**
- Evaluation against rule-based agents (Negamax, Minimax, Block Check)
- Win-rate tracking and move efficiency metrics
- Tested in Connect-4, 6x7 environment

## 🧠 Algorithms Used

### Double DQN

- Reduces Q-learning overestimation by decoupling action selection and evaluation.
- Uses experience replay and target networks for stability.

### PPO

- A policy-gradient method that clips updates to ensure stable learning.
- Implements an Actor-Critic architecture with shared convolutional layers.

## 📊 Evaluation

Agents were evaluated in head-to-head matches and against rule-based opponents. Performance metrics include:

- Win/loss/tie rates
- Average number of moves per game
- Learning curves vs. training steps/episodes

## 📦 Requirements

- Python 3.8+
- PyTorch
- NumPy
- [Kaggle Environments](https://github.com/Kaggle/kaggle-environments)
