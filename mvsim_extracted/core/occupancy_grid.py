"""
Occupancy Grid Mapping (SLAM) module for MVSim.
Enables the robot to actively explore and learn its 2D environment using LiDAR raycasting,
tracking free space, discovered obstacles, and unexplored frontiers.
"""

import math
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict


class OccupancyGrid:
    """
    2D Occupancy Grid Map representing learned spatial geometry.
    Cell values:
      -1  : Unexplored / Unknown (Fog of war)
       0  : Explored Free Space
      100 : Discovered Obstacle / Wall
    """
    CELL_UNKNOWN = -1
    CELL_FREE = 0
    CELL_OBSTACLE = 100

    def __init__(
        self,
        world_width: float = 30.0,
        world_height: float = 24.0,
        resolution: float = 0.20  # meters per grid cell
    ):
        self.world_width = world_width
        self.world_height = world_height
        self.resolution = resolution

        self.cols = int(math.ceil(world_width / resolution))
        self.rows = int(math.ceil(world_height / resolution))

        # Origin corresponds to world (0, 0) in the center
        self.origin_col = self.cols // 2
        self.origin_row = self.rows // 2

        # -1 = unknown, 0 = free, 100 = obstacle
        self.grid = np.full((self.rows, self.cols), self.CELL_UNKNOWN, dtype=np.int8)

        # Precompute disk kernel for obstacle inflation (robot radius ~ 0.45m)
        self.inflation_radius_m = 0.45
        kernel_radius = int(math.ceil(self.inflation_radius_m / self.resolution))
        ksize = 2 * kernel_radius + 1
        self.inflation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        self.total_cells = self.rows * self.cols

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Converts world coordinates (x, y) to grid indices (col, row)."""
        col = int(self.origin_col + round(wx / self.resolution))
        row = int(self.origin_row - round(wy / self.resolution))  # Row increases downward
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Converts grid indices (col, row) to world coordinates (x, y)."""
        wx = (col - self.origin_col) * self.resolution
        wy = (self.origin_row - row) * self.resolution
        return wx, wy

    def is_in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def update_from_lidar(
        self,
        sensor_world_x: float,
        sensor_world_y: float,
        hit_points: List[Tuple[float, float]],
        ranges: List[float],
        ray_angles: List[float],
        sensor_yaw: float,
        max_range: float
    ):
        """
        Updates the occupancy grid based on live LiDAR returns:
        - Carves free space from sensor origin along ray path
        - Marks obstacle cells at verified hit points
        """
        s_col, s_row = self.world_to_grid(sensor_world_x, sensor_world_y)
        if not self.is_in_bounds(s_col, s_row):
            return

        # Fast line rasterization canvas for carving free space
        free_canvas = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # 1. Carve free space along each ray
        n_rays = len(ranges)
        step = 1 if n_rays <= 100 else 2  # Adaptive ray sampling for sub-millisecond mapping speed

        for i in range(0, n_rays, step):
            r = ranges[i]
            # Ray endpoint in world
            ang = sensor_yaw + ray_angles[i]
            dist_endpoint = min(r, max_range)
            ex = sensor_world_x + dist_endpoint * math.cos(ang)
            ey = sensor_world_y + dist_endpoint * math.sin(ang)

            e_col, e_row = self.world_to_grid(ex, ey)
            e_col = max(0, min(self.cols - 1, e_col))
            e_row = max(0, min(self.rows - 1, e_row))

            cv2.line(free_canvas, (s_col, s_row), (e_col, e_row), 1, 1)

        # Apply free cells where not currently confirmed obstacles
        mask_free = (free_canvas == 1) & (self.grid != self.CELL_OBSTACLE)
        self.grid[mask_free] = self.CELL_FREE

        # 2. Mark obstacle hit points
        for (hx, hy) in hit_points:
            h_col, h_row = self.world_to_grid(hx, hy)
            if self.is_in_bounds(h_col, h_row):
                self.grid[h_row, h_col] = self.CELL_OBSTACLE

    def get_inflated_costmap(self) -> np.ndarray:
        """
        Computes a costmap for path planning:
        - Free space: 1
        - Unexplored space: 3 (optimistic traversable with exploration penalty)
        - Inflated obstacle buffer: 50 (steep penalty near walls)
        - Definite Obstacles: 255 (non-traversable)
        """
        costmap = np.full((self.rows, self.cols), 3, dtype=np.uint8)  # Default unknown

        # Free space
        costmap[self.grid == self.CELL_FREE] = 1

        # Obstacles
        obs_mask = (self.grid == self.CELL_OBSTACLE).astype(np.uint8)
        if np.any(obs_mask):
            # Dilate obstacles for robot collision clearance
            inflated_obs = cv2.dilate(obs_mask, self.inflation_kernel)
            costmap[inflated_obs > 0] = 50  # Safety buffer
            costmap[obs_mask > 0] = 255     # Hard obstacle

        return costmap

    def get_exploration_stats(self) -> Dict[str, float]:
        """Calculates exploration coverage and discovered obstacle stats."""
        explored_mask = (self.grid != self.CELL_UNKNOWN)
        explored_count = int(np.count_nonzero(explored_mask))
        obstacle_count = int(np.count_nonzero(self.grid == self.CELL_OBSTACLE))
        free_count = int(np.count_nonzero(self.grid == self.CELL_FREE))

        pct = (explored_count / float(self.total_cells)) * 100.0 if self.total_cells > 0 else 0.0

        return {
            "percent_explored": pct,
            "explored_cells": explored_count,
            "obstacle_cells": obstacle_count,
            "free_cells": free_count,
            "total_cells": self.total_cells
        }
