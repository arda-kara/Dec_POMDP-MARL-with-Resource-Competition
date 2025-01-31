"""
pygame_renderer.py

A modular pygame-based renderer for the ResourceThiefEnv or similar 2D grid environments.
Use this module to create a rendering object, then call .render(env) each step
to visualize the environment's grid and agents.

DEPENDENCIES:
    pip install pygame

USAGE EXAMPLE:
    from envs.resource_thief_env import ResourceThiefEnv
    from pygame_renderer import ResourceThiefRenderer
    import pygame

    def main():
        env = ResourceThiefEnv(grid_size=(5,5), ...)
        renderer = ResourceThiefRenderer(cell_size=60, agent_radius=20, fps=4)
        env.reset()

        running = True
        step_count = 0
        max_steps = 50

        while running:
            step_count += 1
            # 1) Gather events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 2) Decide actions for each agent (random or your policy)
            actions = {ag: 0 for ag in env.agents}  # trivial example

            obs, rew, terms, truncs, infos = env.step(actions)

            # 3) Render environment
            renderer.render(env)

            # 4) Check if done
            if step_count >= max_steps or all(not env._agent_states[ag]["alive"] for ag in env.agents):
                running = False

        renderer.close()

    if __name__ == "__main__":
        main()
"""

import pygame
import sys

class ResourceThiefRenderer:
    """
    A pygame-based renderer class for ResourceThiefEnv (or any grid-based 2D environment).

    ATTRIBUTES:
        cell_size (int):
            Pixel size for each cell in the environment's grid.
        agent_radius (int):
            Radius in pixels for drawing agent circles.
        fps (int):
            Frames per second to control refresh speed in .render().
        screen (pygame.Surface):
            The main pygame display surface (initialized on first render).
        clock (pygame.time.Clock):
            Used to throttle rendering speed.
        window_caption (str):
            Title of the pygame window.
        bg_color (tuple):
            The background color, defaults to black if not changed.

    METHODS:
        render(env):
            Draws env's grid and agents onto a pygame window.
        close():
            Quits pygame and closes the window.
    """

    def __init__(
        self,
        cell_size=60,
        agent_radius=20,
        fps=4,
        window_caption="ResourceThiefEnv",
        bg_color=(0, 0, 0)
    ):
        """
        Constructor for the ResourceThiefRenderer.

        :param cell_size: Pixel size for each grid cell.
        :param agent_radius: Pixel radius for agent circles.
        :param fps: Frames per second for rendering speed.
        :param window_caption: Title of the pygame window.
        :param bg_color: Background color (R,G,B).
        """
        self.cell_size = cell_size
        self.agent_radius = agent_radius
        self.fps = fps
        self.window_caption = window_caption
        self.bg_color = bg_color

        self.screen = None
        self.clock = None
        self.initialized = False

        # Some default colors for drawing
        self.color_grid_line = (150, 150, 150)
        self.color_empty_cell = (200, 200, 200)
        self.color_food_cell = (102, 255, 102)
        self.color_villager = (0, 102, 255)
        self.color_thief = (255, 51, 51)

    def _init_pygame(self, env):
        """
        Internal method to initialize pygame display the first time we render.
        Calculates screen width/height based on env size and self.cell_size.
        """
        pygame.init()
        width = env.width * self.cell_size
        height = env.height * self.cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self.window_caption)
        self.clock = pygame.time.Clock()
        self.initialized = True

    def _draw_grid(self, env):
        """
        Draw the grid cells: color them based on whether there's food or not,
        draw grid lines as well.
        """
        for r in range(env.height):
            for c in range(env.width):
                cell_x = c * self.cell_size
                cell_y = r * self.cell_size
                rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)

                # Food or empty
                if env.grid[r, c] > 0:
                    color = self.color_food_cell
                else:
                    color = self.color_empty_cell

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.color_grid_line, rect, width=1)

    def _draw_agents(self, env):
        """
        Draw living agents as circles plus text for their hunger/inventory/reputation.
        """
        for agent_id in env.agents:
            st = env._agent_states[agent_id]
            if not st["alive"]:
                continue  # skip dead agents

            r, c = st["position"]
            center_x = c * self.cell_size + self.cell_size // 2
            center_y = r * self.cell_size + self.cell_size // 2

            # color by role
            if env.agent_roles[agent_id] == "villager":
                color = self.color_villager
            else:
                color = self.color_thief

            pygame.draw.circle(self.screen, color, (center_x, center_y), self.agent_radius)

            # small text
            font = pygame.font.SysFont(None, 18)
            text_surf = font.render(
                f"H:{st['hunger']} I:{st['inventory']} R:{st['reputation']}",
                True,
                (0, 0, 0)  # black text
            )
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            self.screen.blit(text_surf, text_rect)

    def render(self, env):
        """
        Render the given environment's state onto the pygame window.
        Call this once per environment step.

        :param env: A ResourceThiefEnv (or similar) instance to visualize.
        """
        if not self.initialized:
            self._init_pygame(env)

        # 1) Clear background
        self.screen.fill(self.bg_color)

        # 2) Draw grid
        self._draw_grid(env)

        # 3) Draw agents
        self._draw_agents(env)

        # 4) Flip the display
        pygame.display.flip()

        # 5) Cap the FPS
        if self.clock:
            self.clock.tick(self.fps)

    def close(self):
        """
        Close the pygame window and quit.
        """
        pygame.quit()
        sys.exit()
