import os
from setuptools import setup, find_packages

# Get the long description from the README file
try:
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A library for approximating computationally intractable game theory concepts, focusing on robust equilibria in multi-player games. It leverages techniques like MCTS and Reinforcement Learning to provide scalable solutions where exhaustive enumeration is infeasible."

setup(
    name='robust-equilibrium-approximator',
    version='0.1.0',
    author='AI Developer Team',
    author_email='devteam@example.com', # Placeholder email
    description='A library for approximating computationally intractable game theory concepts, particularly the minimum-gain analogue of strong equilibrium.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-organization/robust-equilibrium-approximator', # Placeholder URL
    packages=find_packages(exclude=['tests*', 'docs*']),
    install_requires=[
        'numpy>=1.20.0',          # Core numerical operations
        'scipy>=1.7.0',           # Scientific computing, e.g., for optimization or statistics
        'stable-baselines3>=2.0.0',# For Reinforcement Learning agents (PPO, A2C, DQN)
        'gymnasium>=0.28.0',      # Standard API for RL environments, often used with Stable Baselines3
        'matplotlib>=3.4.0',      # For plotting and visualization in experiments and results analysis
        'seaborn>=0.11.0',        # For enhanced statistical data visualization
        'pandas>=1.3.0',          # For data handling and analysis in experiments
        'tqdm>=4.60.0',           # For progress bars during simulations and training
        # Add any other specific dependencies as they are identified during development
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License', # Assuming an open-source license like MIT
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='game theory, strong equilibrium, nash equilibrium, approximation, reinforcement learning, mcts, monte carlo tree search, robust equilibrium, multi-agent systems',
    python_requires='>=3.8',
    project_urls={
        'Bug Reports': 'https://github.com/your-organization/robust-equilibrium-approximator/issues',
        'Source': 'https://github.com/your-organization/robust-equilibrium-approximator/',
    },
)