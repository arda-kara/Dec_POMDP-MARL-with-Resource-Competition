"""
resource_thief_env.py

A PettingZoo parallel environment for a resource-thief scenario, focusing on:
  - A large grid with single-food tiles (0 or 1).
  - Villagers can gather from the grid, trade with adjacent villagers, and eat from inventory.
  - Thieves cannot gather from the grid, but must steal from villagers. They can also eat stolen items.
  - Partial local observations plus each agent's own hunger and inventory.
  - End conditions: all agents die or max_steps is reached.
  - No reputation or voting mechanics in this version.

HOW TO USE:
  1) Instantiate this class with environment parameters:
     env = ResourceThiefEnv(grid_size=(15,15), max_food=30, ...)
  2) Reset the environment: obs, info = env.reset()
  3) Step with a dict of agent actions: obs, rewards, dones, truncs, infos = env.step(actions)
  4) Integration with RLlib can be done via an rllib_env_wrapper or PettingZoo wrapper approach.
  5) For a partial local observation, each agent sees a 3x3 area + [my_hunger, my_inv].
     => obs shape ~ (2*3*3 + 2) = 20 if flattening local channels.

ACTION SPACE (0..10):
  0 = do nothing
  1 = move up
  2 = move down
  3 = move left
  4 = move right
  5 = gather (villager-only, if tile has 1 => tile=0, inventory++)
  6 = steal (thief-only, from adjacent or same-cell villager with inv>0 => transfer 1)
  7 = trade (villager-only, with adjacent villager => transfer 1 item)
  8 = placeholder/no-op (unused)
  9 = placeholder/no-op (unused)
  10 = eat (both roles; if inventory>0 => hunger += 15, capped at 100)

OBSERVATION:
  - Flattened local 3x3 region: 2 channels => (food presence 0/1), (occupant role 0=none,1=villager,2=thief)
  - Appended [my_hunger (scaled?), my_inventory].
  => total ~20 dimensions if we treat hunger & inventory as 1D each. Or you can keep them separate.

REWARD:
  - +1 each step for living agents, 0 if dead.
  - Survive as long as possible.

DEPENDENCIES:
  - pettingzoo: for ParallelEnv base
  - gymnasium: for spaces
"""

import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces


class ResourceThiefEnv(ParallelEnv):
    """
    Parallel environment of villagers vs thieves on a 2D grid with single-food tiles.

    KEY POINTS:
      - Single occupant role: villager or thief
      - Single-food tile => grid[r,c] in {0,1}
      - Each agent loses hunger by 1 each step
      - If hunger=0 => agent dies
      - Villagers: gather from tile (action=5), trade with adjacent (7), or eat from inventory (10)
      - Thieves: cannot gather from tile, but can steal from adjacent or same-cell villager (action=6), and eat stolen item (10)
      - Step ends when all dead or max_steps reached
      - Each living agent gets +1 reward each step
      - Observations are partial local (3x3) + [my_hunger, my_inv]
    """

    metadata = {
        "render_modes": ["human", "none"],
        "name": "resource_thief_no_reputation_v0"
    }

    def __init__(
        self,
        grid_size=(15, 15),
        max_food=30,
        food_spawn_prob=0.05,
        num_villagers=5,
        num_thieves=5,
        seed=None,
        view_radius=1,       # partial obs => 3x3 local
        max_steps=200,       # episode length limit
        gather_amount=1,     # each gather => inventory+1
        steal_amount=1,      # each steal => thief inventory+1, victim inventory-1
        trade_amount=1,      # each trade => transfer 1 from actor to neighbor
        hunger_decrement=1,  # each step => hunger minus 1
        eat_replenish=15,    # each item eaten => +15 hunger
        spawn_limit=True     # if True, we do not exceed max_food total across the grid
    ):
        """
        Constructor for ResourceThiefEnv.

        :param grid_size: (height, width) of the grid
        :param max_food: total # of food items allowed on the grid if spawn_limit=True
        :param food_spawn_prob: probability each empty cell spawns 1 food each step
        :param num_villagers: how many villagers to create
        :param num_thieves: how many thieves to create
        :param seed: optional int for random seeding
        :param view_radius: 1 => local 3x3 partial obs
        :param max_steps: forcibly end the episode after this many steps
        :param gather_amount: how many items a villager gathers from a tile (usually 1)
        :param steal_amount: how many items a thief steals from an adjacent or same villager
        :param trade_amount: how many items are transferred by a villager trade action
        :param hunger_decrement: how much hunger is lost each step
        :param eat_replenish: how much hunger is replenished by eating 1 item
        :param spawn_limit: if True, won't spawn new items if total food >= max_food
        """
        super().__init__()

        # Basic config
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.max_food = max_food
        self.food_spawn_prob = food_spawn_prob
        self.num_villagers = num_villagers
        self.num_thieves = num_thieves
        self._seed = seed

        # Observability
        self.view_radius = view_radius
        self.obs_view_size = 2 * self.view_radius + 1  # typically 3
        # Episode management
        self.step_count = 0
        self.max_steps = max_steps

        # Resource/trade/hunger
        self.gather_amount = gather_amount
        self.steal_amount = steal_amount
        self.trade_amount = trade_amount
        self.hunger_decrement = hunger_decrement
        self.eat_replenish = eat_replenish
        self.spawn_limit = spawn_limit

        # Internal structures
        self.agents = []
        self.possible_agents = []
        self.agent_roles = {}
        self._agent_states = {}
        self.render_mode = "none"
        self.grid = None

        # We define an 11-action space (0..10).
        # See docstring for meaning.
        # Observations: partial local 3x3 => 2 channels => 18 + 2 (hunger, inventory) => 20
        self.action_spaces = {}
        self.observation_spaces = {}

        self._init_environment()

    def _init_environment(self):
        """
        Initialize or re-initialize the environment:
          - Create grid with zeros
          - Create agent IDs (villager_x, thief_y)
          - Random spawn positions
          - Create spaces
          - Reset step_count
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        # possible_agents
        self.possible_agents = []
        for i in range(self.num_villagers):
            self.possible_agents.append(f"villager_{i}")
        for j in range(self.num_thieves):
            self.possible_agents.append(f"thief_{j}")

        # start each new episode with all these agents
        self.agents = self.possible_agents[:]

        # grid
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # agent states: (position, inventory, hunger, alive)
        self._agent_states = {}
        for agent_id in self.agents:
            if agent_id.startswith("villager"):
                role = "villager"
            else:
                role = "thief"
            self.agent_roles[agent_id] = role

            # random position
            r = np.random.randint(0, self.height)
            c = np.random.randint(0, self.width)
            self._agent_states[agent_id] = {
                "position": (r, c),
                "inventory": 0,
                "hunger": 100,
                "alive": True
            }

        # define action space
        # 0..10 => 11 actions
        num_actions = 11

        # define observation space
        # local (3x3) => 2 channels => 18
        # +2 => [my_hunger, my_inventory] => total=20
        obs_dim = 2 * (self.obs_view_size**2) + 2

        for agent_id in self.agents:
            self.action_spaces[agent_id] = spaces.Discrete(num_actions)
            self.observation_spaces[agent_id] = spaces.Box(
                low=0, high=255, shape=(obs_dim,), dtype=np.uint8
            )

        self.step_count = 0

    # Required methods for PettingZoo ParallelEnv
    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the environment for a new episode.
        Returns:
          observations: dict of agent_id -> obs (np.array)
          info: None or empty if return_info=False
        """
        if seed is not None:
            np.random.seed(seed)

        self._init_environment()
        # produce initial observations
        observations = self._generate_observations()

        if return_info:
            return observations, {}
        else:
            return observations, None

    def step(self, actions):
        """
        Step the environment with the given dict of actions.
        actions: dict(agent_id -> int)
        Returns:
          obs, rewards, terminations, truncations, infos
        """
        # 1) apply hunger decrement
        for ag in self.agents:
            st = self._agent_states[ag]
            if st["alive"]:
                st["hunger"] -= self.hunger_decrement
                if st["hunger"] <= 0:
                    st["alive"] = False

        # 2) apply each agent's action
        for ag, act in actions.items():
            if self._agent_states[ag]["alive"]:
                self._apply_action(ag, act)

        # 3) spawn food
        self._spawn_food()

        # 4) increment step_count
        self.step_count += 1

        # 5) build returns
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        all_dead = True

        for ag in self.agents:
            st = self._agent_states[ag]
            if st["alive"]:
                rewards[ag] = 1.0  # living => +1
                all_dead = False
                terminations[ag] = False
            else:
                rewards[ag] = 0.0
                terminations[ag] = True  # dead => done for that agent

            truncations[ag] = False
            infos[ag] = {}

        # if we hit max steps or everyone is dead => end
        if self.step_count >= self.max_steps or all_dead:
            for ag in self.agents:
                terminations[ag] = True

        obs = self._generate_observations()
        return obs, rewards, terminations, truncations, infos

    def _apply_action(self, agent_id, action):
        """
        Handle a single agent's action logic:
          1..4 => movement
          5 => gather if villager
          6 => steal if thief
          7 => trade if villager
          10 => eat if inventory>0 => hunger+15
        """
        st = self._agent_states[agent_id]
        r, c = st["position"]
        role = self.agent_roles[agent_id]

        if action == 1:  # up
            nr = max(r - 1, 0)
            st["position"] = (nr, c)
        elif action == 2:  # down
            nr = min(r + 1, self.height - 1)
            st["position"] = (nr, c)
        elif action == 3:  # left
            nc = max(c - 1, 0)
            st["position"] = (r, nc)
        elif action == 4:  # right
            nc = min(c + 1, self.width - 1)
            st["position"] = (r, nc)

        elif action == 5:  # gather
            if role == "villager":
                # if there's 1 food at (r,c), gather it
                if self.grid[r, c] >= self.gather_amount:
                    self.grid[r, c] -= self.gather_amount
                    st["inventory"] += self.gather_amount

        elif action == 6:  # steal
            if role == "thief":
                self._steal_action(agent_id)

        elif action == 7:  # trade
            if role == "villager":
                self._trade_action(agent_id)

        elif action == 10:  # eat
            # both roles can do it
            if st["inventory"] > 0:
                st["inventory"] -= 1
                st["hunger"] = min(st["hunger"] + self.eat_replenish, 100)

        # 0 => do nothing
        # 8,9 => placeholders/no-ops

    def _steal_action(self, thief_id):
        """
        If thief has action=6, attempts to steal from adjacent or same-cell villager with inventory>0.
        Transfers 'steal_amount' from victim to thief if found.
        """
        th_state = self._agent_states[thief_id]
        r, c = th_state["position"]

        neighbors = [
            (r, c),
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1)
        ]
        victim_found = None

        for nr, nc in neighbors:
            if 0 <= nr < self.height and 0 <= nc < self.width:
                # search for any villager
                for other_ag in self.agents:
                    if other_ag == thief_id:
                        continue
                    if self.agent_roles[other_ag] == "villager":
                        st = self._agent_states[other_ag]
                        if st["alive"] and st["position"] == (nr, nc) and st["inventory"] >= self.steal_amount:
                            # steal from them
                            st["inventory"] -= self.steal_amount
                            th_state["inventory"] += self.steal_amount
                            victim_found = other_ag
                            break
                if victim_found is not None:
                    break

    def _trade_action(self, villager_id):
        """
        For villager, action=7 => trade with an adjacent living villager if inventory>0.
        Transfer 'trade_amount' from actor to neighbor.
        """
        v_state = self._agent_states[villager_id]
        if v_state["inventory"] < self.trade_amount:
            return  # can't trade

        r, c = v_state["position"]
        neighbors = [
            (r, c),
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1)
        ]
        for nr, nc in neighbors:
            if 0 <= nr < self.height and 0 <= nc < self.width:
                for other_ag in self.agents:
                    if other_ag == villager_id:
                        continue
                    if self.agent_roles[other_ag] == "villager":
                        st = self._agent_states[other_ag]
                        if st["alive"] and st["position"] == (nr, nc):
                            # do 1 item transfer
                            v_state["inventory"] -= self.trade_amount
                            st["inventory"] += self.trade_amount
                            return  # only trade once this step

    def _spawn_food(self):
        """
        Spawns single-food tiles in empty cells with probability food_spawn_prob,
        if spawn_limit is True, won't exceed 'max_food' total across the grid.
        """
        if self.spawn_limit:
            current_food_count = np.sum(self.grid)
            if current_food_count >= self.max_food:
                return  # no spawn if we are at limit

        # attempt spawn
        for rr in range(self.height):
            for cc in range(self.width):
                if self.grid[rr, cc] == 0:
                    if np.random.random() < self.food_spawn_prob:
                        if not self.spawn_limit:
                            self.grid[rr, cc] = 1
                        else:
                            # check if under the limit
                            current_food = np.sum(self.grid)
                            if current_food < self.max_food:
                                self.grid[rr, cc] = 1

    def _generate_observations(self):
        """
        Generate observations for each agent:
          - local 3x3 slice => 2 channels => 18
          - plus [my_hunger, my_inventory] => total 20
        Return dict(agent_id -> np.array of shape (20,))
        """
        obs_dict = {}
        for ag in self.agents:
            st = self._agent_states[ag]
            if not st["alive"]:
                # dead agent => zero obs
                obs_dict[ag] = np.zeros((2 * self.obs_view_size**2 + 2,), dtype=np.uint8)
                continue

            # local 3x3 channel
            local_food_map = np.zeros((self.obs_view_size, self.obs_view_size), dtype=np.uint8)
            local_role_map = np.zeros((self.obs_view_size, self.obs_view_size), dtype=np.uint8)

            r, c = st["position"]
            # e.g. if view_radius=1 => [r-1..r+1], same for c
            rmin, rmax = r - self.view_radius, r + self.view_radius
            cmin, cmax = c - self.view_radius, c + self.view_radius

            for rr in range(rmin, rmax+1):
                for cc in range(cmin, cmax+1):
                    local_r = rr - rmin
                    local_c = cc - cmin
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        # food presence
                        local_food_map[local_r, local_c] = min(1, self.grid[rr, cc])

                        # occupant role
                        occupant_val = 0
                        # find if a living agent is at (rr, cc)
                        for other_ag in self.agents:
                            if self._agent_states[other_ag]["alive"]:
                                (or_, oc_) = self._agent_states[other_ag]["position"]
                                if (or_, oc_) == (rr, cc):
                                    if self.agent_roles[other_ag] == "villager":
                                        occupant_val = 1
                                    else:
                                        occupant_val = 2
                                    break
                        local_role_map[local_r, local_c] = occupant_val

            # Flatten local 3x3 => shape= 2*(3x3)=18
            flattened_local = np.concatenate([
                local_food_map.flatten(),
                local_role_map.flatten()
            ]).astype(np.uint8)

            # add [my_hunger, my_inventory]
            # clamp them to 255 just for safety in np.uint8
            hunger_val = min(st["hunger"], 255)
            inv_val = min(st["inventory"], 255)
            appended = np.array([hunger_val, inv_val], dtype=np.uint8)

            full_obs = np.concatenate([flattened_local, appended], axis=0)
            obs_dict[ag] = full_obs
        return obs_dict

    def render(self):
        """
        If self.render_mode=='human', print a text-based representation of the grid and agent states.
        """
        if self.render_mode == "human":
            print("=== ResourceThiefEnv Render ===")
            print(f"Step: {self.step_count}/{self.max_steps}")
            print("Grid:")
            print(self.grid)
            for ag in self.agents:
                st = self._agent_states[ag]
                print(f"  {ag} => Pos={st['position']}, "
                      f"Hunger={st['hunger']}, Inv={st['inventory']}, "
                      f"Alive={st['alive']}")
        # else: do nothing

    def close(self):
        """
        A no-op close method for PettingZoo compliance.
        """
        pass

    # PettingZoo recommended overrides
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
