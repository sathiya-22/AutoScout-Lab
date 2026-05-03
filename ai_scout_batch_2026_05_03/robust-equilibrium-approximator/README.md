# Robust Equilibrium Approximator

## Introduction
The Robust Equilibrium Approximator is a Python library designed to tackle the computational challenges of finding approximate solutions for computationally intractable game theory concepts, such as the minimum-gain analogue of the strong equilibrium. It provides a modular, scalable framework leveraging advanced AI techniques to manage the combinatorial explosion inherent in large games.

## The Problem: Intractability of Strong Equilibria
Traditional game theory concepts, like the minimum-gain analogue of the strong equilibrium, which aims to minimize the smallest gain achievable by any deviating coalition, pose significant computational hurdles. For practical game sizes, direct computation is impossible due to the exponential growth in the number of potential coalitions and their deviation incentives. This combinatorial explosion makes exhaustive enumeration infeasible, rendering exact solutions unattainable even for relatively small numbers of players.

## The Solution: Approximate Robust Equilibria
This library offers an innovative approach to overcome this intractability by focusing on approximate solutions. It implements a hybrid methodology combining techniques like Monte Carlo Tree Search (MCTS) and Reinforcement Learning (RL) to strategically explore the vast space of coalition deviations. The goal is to identify high-impact coalitions and learn strategies that minimize their worst-case gains without requiring exhaustive enumeration. The library provides flexible, pluggable interfaces for defining custom game utility functions and coalition formation rules, empowering users to apply scalable approximate solver algorithms to games with a large number of players through sampling and learned heuristics.

## Core Principles
The Robust Equilibrium Approximator is built upon the following foundational principles:

1.  **Modularity:** Emphasizes clear separation of concerns for game definition, approximation algorithms, and coalition logic.
2.  **Pluggable Interfaces:** Provides abstract interfaces that allow users to define custom games, utility functions, and coalition formation rules, ensuring high extensibility.
3.  **Approximation Focus:** Leverages sampling and learned heuristics to efficiently navigate and solve problems that would otherwise be subject to combinatorial explosion.

## Architecture Overview
The library's design is structured into several key components, each with a distinct role in facilitating the approximation of robust equilibria:

### 1. `robust_equilibrium_approximator/core/`
This foundational module defines the essential interfaces and data structures.
*   **`interfaces.py`**: Contains abstract base classes like `IGame`, `IPlayer`, `IAction`, `IUtilityFunction`, `IApproximator`, and `ICoalitionRule`. These interfaces form the backbone for extending and customizing the library.
*   **`game_state.py`**: Models the current state of a game, including player actions and available information.
*   **`strategy_profile.py`**: Manages and facilitates the evaluation of strategies chosen by all participating players.

### 2. `robust_equilibrium_approximator/games/`
This module houses concrete implementations of various games, all adhering to the `core` interfaces.
*   **`base.py`**: Provides abstract base classes and common utilities for game implementations.
*   **`example_min_gain_game.py`**: Offers a practical example demonstrating how to define a game relevant to the minimum-gain strong equilibrium problem, enabling rapid prototyping and testing.

### 3. `robust_equilibrium_approximator/coalitions/`
Dedicated to the complex dynamics of coalition formation and evaluation.
*   **`rules.py`**: Defines various algorithms and criteria for generating potential coalitions.
*   **`manager.py`**: Responsible for dynamically forming coalitions, validating their adherence to specified rules, and critically, evaluating their deviation incentives and potential gains. This component is vital for identifying the 'worst-case' deviating coalition that the equilibrium approximator aims to mitigate.

### 4. `robust_equilibrium_approximator/approximators/`
The intellectual core of the library, implementing sophisticated algorithms for approximate equilibrium finding.

*   **`mcts_solver.py`**: Implements a Monte Carlo Tree Search (MCTS) approach. MCTS explores the vast space of strategy profiles and coalition deviations through simulated game play. Nodes represent game states or partial strategy profiles, and actions involve selecting player strategies or evaluating coalition deviations. The 'reward' mechanism for MCTS is meticulously tailored to minimize the maximum gain achievable by any deviating coalition, effectively guiding the search towards robust equilibrium strategies.
*   **`rl_agent_solver.py`**: Contains implementations of Reinforcement Learning (RL) agents, potentially utilizing frameworks like Stable Baselines3 for algorithms such as PPO, A2C, or DQN. The RL agent learns an optimal strategy for all players by interacting with the game environment. The reward function is carefully crafted to penalize high-gain coalition deviations, thereby training the agent to discover strategies that intrinsically minimize the worst-case deviation gain.
*   **`hybrid_solver.py`**: Explores synergistic combinations of MCTS and RL. For instance, MCTS could be employed to strategically identify high-impact coalition deviations, with RL agents then learning optimal responses within these identified subgames. Alternatively, a pre-trained RL policy or value function could inform MCTS tree expansion for more efficient and focused exploration.

### 5. `robust_equilibrium_approximator/utils/`
Provides a collection of common utility functions to support various components of the library.
*   **`logging_config.py`**: For standardized and configurable logging across the library.
*   **`metrics.py`**: For quantitative evaluation of approximation quality, including comparison against known benchmarks for small games or calculation of error bounds.
*   **`heuristics.py`**: General-purpose heuristic functions used in sampling strategies or pruning within the approximators and coalition manager.

### 6. `robust_equilibrium_approximator/experiments/`
A robust framework for systematically running, evaluating, and comparing the performance of different approximators across diverse game instances.
*   **`runner.py`**: Orchestrates the execution of experiments.
*   **`results_analyzer.py`**: Provides tools for analyzing, visualizing, and interpreting experimental outcomes, aiding in understanding the strengths and weaknesses of each approximation method.

## Addressing Intractability & Scalability
The library directly addresses the computational intractability and scalability challenges by:
*   **Sampling:** Leveraging techniques inherent in MCTS simulations and RL experience collection to avoid the need for exhaustive enumeration of strategy profiles and coalitions.
*   **Learned Heuristics:** Utilizing knowledge derived from RL value functions and policies, along with domain-specific heuristics, to guide search and decision-making processes efficiently.
*   **Modular Design:** Facilitating the seamless integration of new, more efficient algorithms and data structures as research evolves, ensuring the library remains adaptable and performant for increasingly larger game scales.
*   **Pruning:** The `coalitions/manager.py` can incorporate pruning heuristics to limit the exploration of less impactful coalitions, focusing computational resources where they matter most.

## Getting Started
*(Placeholder for future content: Instructions on how to install the library, define a custom game, and run an approximation solver.)*

## Contributing
*(Placeholder for future content: Guidelines for contributing to the project.)*

## License
*(Placeholder for future content: Information about the project's licensing.)*