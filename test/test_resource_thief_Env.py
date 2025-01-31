"""
test_resource_thief_env.py

A unittest-based test suite for the ResourceThiefEnv environment.
We test:
  - Environment creation, reset
  - Step function (hunger decrement, basic do-nothing action)
  - Gathering logic (villager gather from grid)
  - Stealing logic (thief steals from villager)
  - Trading logic (villager trades with another villager)
  - Eating logic (any agent with inventory eats => +15 hunger)
  - Food spawning
  - Episode termination (max_steps or all agents dead)
"""

import unittest
import numpy as np
from envs.resource_thief_env import ResourceThiefEnv


class TestResourceThiefEnv(unittest.TestCase):
    def setUp(self):
        """
        Creates a small environment for quick testing.
        We'll reduce the grid size for speed and clarity,
        and keep spawn probabilities or max_food small.
        """
        self.env_config = {
            "grid_size": (5, 5),
            "max_food": 5,
            "food_spawn_prob": 0.2,
            "num_villagers": 2,
            "num_thieves": 2,
            "seed": 42,
            "view_radius": 1,
            "max_steps": 10,
            "gather_amount": 1,
            "steal_amount": 1,
            "trade_amount": 1,
            "hunger_decrement": 1,
            "eat_replenish": 15,
            "spawn_limit": True
        }
        self.env = ResourceThiefEnv(**self.env_config)

    def test_env_creation_and_reset(self):
        """
        Test environment creation and a reset to see if initial states are properly set.
        """
        obs, info = self.env.reset()
        # We expect possible_agents = ["villager_0", "villager_1", "thief_0", "thief_1"]
        expected_ids = set(["villager_0", "villager_1", "thief_0", "thief_1"])
        self.assertEqual(set(self.env.agents), expected_ids)

        # Check observation structure
        for ag in self.env.agents:
            self.assertIn(ag, obs)
            self.assertEqual(obs[ag].shape[0], 20)  # local obs= 3x3*2 +2 => 20

        # Check initial step_count=0, grid of zeros
        self.assertEqual(self.env.step_count, 0)
        self.assertTrue(np.all(self.env.grid == 0))

    def test_step_do_nothing(self):
        """
        Step environment with all do-nothing actions (0).
        Checks hunger decrement, reward structure, etc.
        """
        self.env.reset()
        all_actions = {ag: 0 for ag in self.env.agents}
        obs, rewards, term, trunc, infos = self.env.step(all_actions)

        # Check each living agent lost 1 hunger, got +1 reward
        for ag in self.env.agents:
            st = self.env._agent_states[ag]
            if st["alive"]:
                self.assertEqual(st["hunger"], 99)  # started at 100, -1
                self.assertEqual(rewards[ag], 1.0)
                self.assertFalse(term[ag])
                self.assertFalse(trunc[ag])

        self.assertEqual(self.env.step_count, 1)

    def test_gather_action(self):
        """
        Place a villager on a cell with food=1, confirm gather works => tile=0, inventory++
        """
        self.env.reset()
        # Place 'villager_0' on (0,0), put grid(0,0)=1
        self.env._agent_states["villager_0"]["position"] = (0, 0)
        self.env.grid[0, 0] = 1

        # gather => action=5
        actions = {ag: 0 for ag in self.env.agents}  # default do nothing
        actions["villager_0"] = 5

        _, rewards, _, _, _ = self.env.step(actions)

        # villager_0 should have inventory=1, tile =>0
        st_v0 = self.env._agent_states["villager_0"]
        self.assertEqual(st_v0["inventory"], 1)
        self.assertEqual(self.env.grid[0, 0], 0)

        # Check reward => +1 if alive
        self.assertEqual(rewards["villager_0"], 1.0)

    def test_steal_action(self):
        """
        Thief steals from an adjacent or same-cell villager with inventory>0 => transfer 1
        """
        self.env.reset()
        # Force positions
        self.env._agent_states["villager_0"]["position"] = (0, 0)
        self.env._agent_states["thief_0"]["position"] = (0, 1)

        # Give villager_0 inventory=2
        self.env._agent_states["villager_0"]["inventory"] = 2

        # thief_0 => action=6 => steal
        actions = {ag: 0 for ag in self.env.agents}
        actions["thief_0"] = 6

        obs, rewards, terms, truncs, infos = self.env.step(actions)
        # th0 should have stolen 1 from v0
        st_th0 = self.env._agent_states["thief_0"]
        st_v0 = self.env._agent_states["villager_0"]
        self.assertEqual(st_th0["inventory"], 1)
        self.assertEqual(st_v0["inventory"], 1)

    def test_trade_action(self):
        """
        If villager_0 trades with villager_1 in adjacent cells => transfer 1 item
        """
        self.env.reset()
        # create 2 villagers, place them adjacent
        self.env._agent_states["villager_0"]["position"] = (0, 0)
        self.env._agent_states["villager_1"]["position"] = (0, 1)

        # v0 has inventory=2, v1=0
        self.env._agent_states["villager_0"]["inventory"] = 2

        # v0 => trade => action=7
        actions = {ag: 0 for ag in self.env.agents}
        actions["villager_0"] = 7

        self.env.step(actions)

        # expect v0 => inventory=1, v1=> inventory=1
        st_v0 = self.env._agent_states["villager_0"]
        st_v1 = self.env._agent_states["villager_1"]
        self.assertEqual(st_v0["inventory"], 1)
        self.assertEqual(st_v1["inventory"], 1)

    def test_eat_action(self):
        """
        Both villagers and thieves can do action=10 => eat => hunger +15, inv-1
        """
        self.env.reset()
        # let villager_0 => inventory=2, hunger=80
        self.env._agent_states["villager_0"]["inventory"] = 2
        self.env._agent_states["villager_0"]["hunger"] = 80

        # let thief_0 => inventory=1, hunger=50
        self.env._agent_states["thief_0"]["inventory"] = 1
        self.env._agent_states["thief_0"]["hunger"] = 50

        # actions => both do 10 => eat
        actions = {ag: 0 for ag in self.env.agents}
        actions["villager_0"] = 10
        actions["thief_0"] = 10

        self.env.step(actions)

        st_v0 = self.env._agent_states["villager_0"]
        st_th0 = self.env._agent_states["thief_0"]

        # each used 1 item => inventory--
        self.assertEqual(st_v0["inventory"], 1)
        self.assertEqual(st_th0["inventory"], 0)

        # hunger => +15 minus 1 from the step => net +14 from original
        # villager_0 => 80 => +15=95 => minus 1 step=94 (since step function does hunger-- first)
        # but let's confirm the order
        # Code does hunger decrement first => so from 80 =>79, then eat =>79+15=94
        self.assertEqual(st_v0["hunger"], 94)

        # similarly thief => 50 => 49 after step decrement => +15 =>64
        self.assertEqual(st_th0["hunger"], 64)

    def test_food_spawning(self):
        """
        Confirm that each step, some empty cells might become 1 if random < food_spawn_prob,
        up to max_food if spawn_limit is True.
        We'll do multiple steps and see if the grid changes from 0 to 1.
        Because random, we can't guarantee, but we can check if sum > 0 eventually.
        """
        self.env_config["max_food"] = 3
        # re-create env with a small max_food
        env = ResourceThiefEnv(**self.env_config)
        env.reset()

        # ensure we start with all zeros => sum=0
        self.assertEqual(np.sum(env.grid), 0)

        steps_to_try = 15
        found_spawning = False
        for _ in range(steps_to_try):
            # do a do-nothing step
            actions = {ag: 0 for ag in env.agents}
            env.step(actions)
            # check the sum of the grid
            s = np.sum(env.grid)
            if s > 0:
                found_spawning = True
                break

        # it is random, but over 15 steps with prob=0.2, likely we get some
        # but possible no spawn if random is unlucky
        # We'll just check if spawn is possible => found_spawning might be True typically
        # We'll not strictly require it to be True to pass the test, but let's do so:
        # If it fails occasionally from randomness, consider seeding or a bigger step
        self.assertTrue(found_spawning, "No food spawn after 15 steps (random chance). Possibly unlucky.")


    def test_episode_termination(self):
        """
        For a short max_steps=3, see if environment terminates after 3 steps or if all die earlier.
        """
        short_config = self.env_config.copy()
        short_config["max_steps"] = 3
        env = ResourceThiefEnv(**short_config)
        obs, info = env.reset()

        done_count = 0
        for step_i in range(10):  # 3 steps is the limit
            acts = {ag: 0 for ag in env.agents}
            obs, rew, term, trunc, inf = env.step(acts)
            # after step, see if we ended
            done_agents = [a for a in term if term[a]]
            done_count += len(done_agents)
            if all(term[a] for a in env.agents):
                # all done or step_count>3
                self.assertLessEqual(env.step_count, 3)
                break

        # We confirm it didn't exceed 3 steps or infinite loop
        self.assertLessEqual(env.step_count, 3)


if __name__ == "__main__":
    unittest.main()
