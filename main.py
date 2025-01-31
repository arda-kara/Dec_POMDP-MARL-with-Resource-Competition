# main.py

from train_marl import run_marl_training

def main():
    env_config = {
        "grid_size": (10,10),
        "max_food": 20,
        "food_spawn_prob": 0.2,
        "num_villagers": 3,
        "num_thieves": 3,
        "max_steps": 100,
        # etc.
    }

    # Let's say we do 50 iterations
    algo = run_marl_training(
        num_iterations=50,
        env_config=env_config,
        checkpoint_freq=10
    )

if __name__ == "__main__":
    main()
