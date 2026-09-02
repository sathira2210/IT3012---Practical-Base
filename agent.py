# agent.py
import random
from collections import deque


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        _ = percept.get('agent_pos')
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """React to immediate percepts without memory."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            return 'Up'
        if percept.get('wall_ahead', False):
            return 'Left'
        return 'Right'


class ModelBasedAgent:
    """Maintain simple memory so it can avoid repeating the same failed action."""

    def __init__(self):
        self.last_percept = None
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if self.last_percept == percept and self.last_action is not None:
            alternatives = [action for action in ['Up', 'Down', 'Left', 'Right'] if action != self.last_action]
            action = alternatives[0] if alternatives else 'Up'
        elif percept.get('wall_ahead', False):
            action = 'Left'
        elif percept.get('food_here', False):
            action = 'Up'
        else:
            action = 'Right'

        self.last_percept = percept
        self.last_action = action
        return action


class SearchAgent:
    """Use breadth-first search to find the shortest path through a static grid."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start: tuple[int, int] = (start_pos[0], start_pos[1])
        goal: tuple[int, int] = (goal_pos[0], goal_pos[1])
        width, height = grid_size
        walls = set(walls)

        if start == goal:
            return []
        if goal in walls:
            return None

        queue: deque[tuple[int, int]] = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        actions: dict[tuple[int, int], str] = {}
        moves = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        while queue:
            current = queue.popleft()
            if current == goal:
                break

            x, y = current
            for action, (dx, dy) in moves:
                nxt = (x + dx, y + dy)
                if (
                    0 <= nxt[0] < width
                    and 0 <= nxt[1] < height
                    and nxt not in walls
                    and nxt not in parents
                ):
                    parents[nxt] = current
                    actions[nxt] = action
                    queue.append(nxt)

        if goal not in parents:
            return None

        path: list[str] = []
        node: tuple[int, int] = goal
        while True:
            parent = parents[node]
            if parent is None:
                break
            path.append(actions[node])
            node = parent
        path.reverse()
        return path