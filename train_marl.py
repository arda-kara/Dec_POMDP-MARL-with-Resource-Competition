"""
train_marl.py

A module that trains a multi-agent PPO policy (villager vs thief) on ResourceThiefEnv
using RLlib with minimized console spam and short episodes to avoid NaN rewards.

KEY CHANGES:
  - Shorter episodes (env_config["max_steps"] = 50) so episodes finish frequently,
    thus RLlib can compute episode_reward_mean right away (reducing NaN).
  - Lower RLlib log_level to "WARN" to avoid huge dictionary prints each iteration.
  - Structured iteration logs for reward stats.

USAGE (in your main.py):
-----------------------------------------------------------------
from train_marl import run_marl_training

def main():
    env_config = {
        "grid_size": (8, 8),
        "max_food": 10,
        "food_spawn_prob": 0.1,
        "num_villagers": 2,
        "num_thieves": 2,
        "max_steps": 50,  # short episodes => quick finishes
    }
    run_marl_training(num_iterations=30, env_config=env_config)

if __name__ == "__main__":
    main()
-----------------------------------------------------------------
"""

import time
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from envs.rllib_env_wrapper import make_env


def run_marl_training(
    num_iterations=50,
    env_name="ResourceThiefEnv",
    env_config=None,
    stop_when_reward=None,
    checkpoint_freq=5
):
    """
    Train multi-agent PPO with separate policies for 'villager' and 'thief',
    with short episodes and minimized console logs.

    :param num_iterations: number of RLlib training iterations
    :param env_name: name for environment (registered in tune)
    :param env_config: dict of env parameters (ensuring max_steps=50 or so)
    :param stop_when_reward: optional float, early stop if avg reward >= threshold
    :param checkpoint_freq: checkpoint frequency in iterations
    :return: trained RLlib algorithm object
    """
    if env_config is None:
        env_config = {}

    # 1) Make sure we have a short max_steps to avoid long episodes => no NaN
    if "max_steps" not in env_config:
        env_config["max_steps"] = 50
    elif env_config["max_steps"] > 200:
        print(
            f"Warning: 'max_steps' in env_config is {env_config['max_steps']}, "
            f"which may cause infrequent episode completions => NaN results!"
        )

    ray.init(ignore_reinit_error=True)

    # 2) Register environment
    tune.register_env(env_name, lambda cfg: make_env(cfg))

    # 3) Define multi-agent policies
    policies = {
        "villager_policy": (None, None, None, {}),
        "thief_policy": (None, None, None, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if "villager" in agent_id:
            return "villager_policy"
        else:
            return "thief_policy"

    # 4) Build PPO config, set log_level="WARN" to reduce console spam
    ppo_config = (
        PPOConfig()
        .debugging(log_level="WARN")  # reduce the large dictionary prints
        # older RLlib approach
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0)  # single worker
        .environment(env=env_name, env_config=env_config)
        .framework("torch")
        # Basic MLP for partial obs
        .training(
            model={
                "conv_filters": None,
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
            train_batch_size=400,
            gamma=0.99,
            lr=1e-3
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["villager_policy", "thief_policy"]
        )
    )

    algo = ppo_config.build()

    print(f"\nStarting multi-agent PPO training for {num_iterations} iterations.")
    print(f"Environment Config: {env_config}\n")

    start_time = time.time()

    for i in range(num_iterations):
        result = algo.train()

        iter_number = result["training_iteration"]
        reward_mean = result.get("episode_reward_mean", float("nan"))
        reward_min = result.get("episode_reward_min", float("nan"))
        reward_max = result.get("episode_reward_max", float("nan"))
        timesteps_total = result.get("timesteps_total", 0)
        time_total_s = result.get("time_total_s", 0.0)

        # We do a simpler structured print
        print(
            f"[Iter {iter_number:03d}/{num_iterations:03d}] "
            f"Reward(Min/Mean/Max): "
            f"{reward_min:.2f}/{reward_mean:.2f}/{reward_max:.2f}  |  "
            f"Timesteps: {timesteps_total}  |  "
            f"ElapsedTime: {time_total_s:.1f}s"
        )

        # checkpoint as needed
        if checkpoint_freq > 0 and (i % checkpoint_freq == 0):
            chkpt_path = algo.save()
            print(f"  >> Saved checkpoint to: {chkpt_path}")

        # optional early stop
        if stop_when_reward is not None and not np.isnan(reward_mean):
            if reward_mean >= stop_when_reward:
                print(
                    f"Stopping early at iter {iter_number} "
                    f"because reward_mean >= {stop_when_reward}"
                )
                break

    total_time = time.time() - start_time
    final_chkpt = algo.save()
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    print(f"Final checkpoint saved at: {final_chkpt}\n")

    ray.shutdown()
    return algo
