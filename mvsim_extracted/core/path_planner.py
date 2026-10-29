"""
Global A* Path Planner with Line-of-Sight Trajectory Smoothing.
Plans collision-free routes through explored and unexplored terrain,
navigating through warehouse aisles and avoiding obstacles.
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Optional
from .occupancy_grid import OccupancyGrid


class AStarPlanner:
    """
    A* Global Planner operating on the learned OccupancyGrid costmap.
    """
    # 8-connectivity motion model: (dx, dy, step_cost)
    NEIGHBORS = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (-1, -1, 1.414)
    ]

    def __init__(self, clearance_weight: float = 3.5):
        self.clearance_weight = clearance_weight

    def plan(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
        occ_grid: OccupancyGrid
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plans a collision-free path from start_world to goal_world.
        Returns a smoothed list of waypoints in world coordinates [(x, y), ...].
        """
        costmap = occ_grid.get_inflated_costmap()

        start_col, start_row = occ_grid.world_to_grid(*start_world)
        goal_col, goal_row = occ_grid.world_to_grid(*goal_world)

        # Ensure start and goal are within bounds
        if not occ_grid.is_in_bounds(start_col, start_row) or not occ_grid.is_in_bounds(goal_col, goal_row):
            return None

        # If goal is inside a hard obstacle or boundary, find nearest traversable neighbor
        if costmap[goal_row, goal_col] >= 200:
            found_alt = False
            for r in range(1, 8):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nc, nr = goal_col + dx, goal_row + dy
                        if occ_grid.is_in_bounds(nc, nr) and costmap[nr, nc] < 200:
                            goal_col, goal_row = nc, nr
                            found_alt = True
                            break
                    if found_alt:
                        break
                if found_alt:
                    break
            if not found_alt:
                return None

        # Priority queue for A* search: (f_score, counter, (col, row))
        open_set = []
        counter = 0

        g_score = { (start_col, start_row): 0.0 }
        h_start = math.hypot(goal_col - start_col, goal_row - start_row)
        heapq.heappush(open_set, (h_start, counter, (start_col, start_row)))

        came_from = {}
        visited = set()

        max_iterations = 8000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, _, (c, r) = heapq.heappop(open_set)

            if (c, r) == (goal_col, goal_row):
                # Reconstruct raw grid path
                raw_path = [(c, r)]
                curr = (c, r)
                while curr in came_from:
                    curr = came_from[curr]
                    raw_path.append(curr)
                raw_path.reverse()

                # Convert grid cells to world coordinates
                world_path = [occ_grid.grid_to_world(col, row) for (col, row) in raw_path]

                # Smooth the path with line-of-sight shortcutting
                smoothed_path = self.smooth_path(world_path, costmap, occ_grid)
                return smoothed_path

            if (c, r) in visited:
                continue
            visited.add((c, r))

            current_g = g_score[(c, r)]

            for dx, dy, step_dist in self.NEIGHBORS:
                nc, nr = c + dx, r + dy

                if not occ_grid.is_in_bounds(nc, nr):
                    continue

                cell_cost = int(costmap[nr, nc])
                if cell_cost >= 200:
                    continue  # Impassable obstacle

                # Cost model: step_dist * traversability multiplier + clearance penalty
                # cell_cost: 1 for free, 3 for unknown, 50 for safety buffer
                move_cost = step_dist * (1.0 + (cell_cost * 0.15))
                tentative_g = current_g + move_cost

                neighbor = (nc, nr)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (c, r)
                    g_score[neighbor] = tentative_g
                    h = math.hypot(goal_col - nc, goal_row - nr)
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h, counter, neighbor))

        return None

    def smooth_path(
        self,
        world_path: List[Tuple[float, float]],
        costmap: np.ndarray,
        occ_grid: OccupancyGrid
    ) -> List[Tuple[float, float]]:
        """
        Applies line-of-sight shortcutting (string-pulling) to turn jagged grid paths
        into smooth, direct polygonal navigation waypoints.
        """
        if len(world_path) <= 2:
            return world_path

        smoothed = [world_path[0]]
        current_idx = 0

        while current_idx < len(world_path) - 1:
            # Look ahead as far as possible with clear line of sight
            furthest_valid_idx = current_idx + 1
            for test_idx in range(len(world_path) - 1, current_idx + 1, -1):
                if self._has_line_of_sight(world_path[current_idx], world_path[test_idx], costmap, occ_grid):
                    furthest_valid_idx = test_idx
                    break

            smoothed.append(world_path[furthest_valid_idx])
            current_idx = furthest_valid_idx

        return smoothed

    @staticmethod
    def _has_line_of_sight(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        costmap: np.ndarray,
        occ_grid: OccupancyGrid
    ) -> bool:
        """Checks if a straight ray between p1 and p2 is collision-free in the costmap."""
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(2, int(dist / (occ_grid.resolution * 0.5)))

        for s in range(steps + 1):
            t = s / float(steps)
            wx = p1[0] + t * (p2[0] - p1[0])
            wy = p1[1] + t * (p2[1] - p1[1])

            col, row = occ_grid.world_to_grid(wx, wy)
            if not occ_grid.is_in_bounds(col, row):
                return False

            if costmap[row, col] >= 50:  # Respects obstacle and safety buffer clearance
                return False

        return True
