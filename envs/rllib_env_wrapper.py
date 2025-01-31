"""
rllib_env_wrapper.py

Provides a factory function `make_env(config)` for RLlib to create a PettingZooEnv-
wrapped ResourceThiefEnv (parallel environment) with the updated mechanics:
  - Larger grid
  - Single-food tiles
  - Villagers can gather, trade, eat
  - Thieves can steal, eat
  - No voting or reputation
  - Partial local observations

Usage in RLlib:
  - from ray import tune
  - from rllib_env_wrapper import make_env
  - tune.register_env("ResourceThiefEnv", make_env)
  - Then specify env="ResourceThiefEnv" in your PPOConfig or register via env_creator.
"""

import numpy as np

from pettingzoo.utils.conversions import parallel_to_aec
from ray.rllib.env import PettingZooEnv

# Import your updated ResourceThiefEnv with no voting/reputation
from envs.resource_thief_env import ResourceThiefEnv


def make_env(config=None):
    """
    Factory function to instantiate ResourceThiefEnv and wrap it for RLlib.

    :param config: dictionary of environment parameters from RLlib, e.g.:
        {
          "grid_size": (15,15),
          "max_food": 30,
          "food_spawn_prob": 0.05,
          "num_villagers": 5,
          "num_thieves": 5,
          "seed": 42,
          "view_radius": 1,
          "max_steps": 200,
          "gather_amount": 1,
          "steal_amount": 1,
          "trade_amount": 1,
          "hunger_decrement": 1,
          "eat_replenish": 15,
          "spawn_limit": True
        }

    :return: A PettingZooEnv wrapping the parallel environment for RLlib.
    """
    if config is None:
        config = {}

    # Extract environment params from config or defaults
    grid_size = config.get("grid_size", (5, 5))
    max_food = config.get("max_food", 10)
    food_spawn_prob = config.get("food_spawn_prob", 0.1)
    num_villagers = config.get("num_villagers", 2)
    num_thieves = config.get("num_thieves", 1)
    seed = config.get("seed", None)
    view_radius = config.get("view_radius", 1)
    max_steps = config.get("max_steps", 100)
    gather_amount = config.get("gather_amount", 1)
    steal_amount = config.get("steal_amount", 1)
    trade_amount = config.get("trade_amount", 1)
    hunger_decrement = config.get("hunger_decrement", 1)
    eat_replenish = config.get("eat_replenish", 15)
    spawn_limit = config.get("spawn_limit", True)

    # Create the parallel environment
    parallel_env = ResourceThiefEnv(
        grid_size=grid_size,
        max_food=max_food,
        food_spawn_prob=food_spawn_prob,
        num_villagers=num_villagers,
        num_thieves=num_thieves,
        seed=seed,
        view_radius=view_radius,
        max_steps=max_steps,
        gather_amount=gather_amount,
        steal_amount=steal_amount,
        trade_amount=trade_amount,
        hunger_decrement=hunger_decrement,
        eat_replenish=eat_replenish,
        spawn_limit=spawn_limit
    )

    # Convert parallel -> AEC env if needed (some RLlib versions prefer AEC)
    aec_env = parallel_to_aec(parallel_env)

    # Wrap in PettingZooEnv for RLlib multi-agent
    return PettingZooEnv(aec_env)
