"""
Artificial Potential Field (APF) obstacle avoidance controller.
Directly ported and extracted from:
- examples_python/simple-obstacle-avoidance.py
"""

import math
from typing import List, Tuple, Optional
from .kinematics import VehicleBase, Twist2D, Pose2D
from .sensors import LaserScanner


class APFObstacleAvoidance:
    """
    Artificial Potential Field (APF) controller ported directly from MVSim's
    examples_python/simple-obstacle-avoidance.py.

    Combines:
    1. Attractive force toward the goal target
    2. Repulsive force inversely proportional to distance squared (1 / r^2) from all lidar scan rays
    """
    def __init__(
        self,
        v_max: float = 1.2,
        w_max: float = 2.0,
        target_attractive_force: float = 8.0,
        obstacle_repulsive_weight: float = 0.8,
        influence_distance: float = 3.5,
        goal_tolerance: float = 0.3
    ):
        self.v_max = v_max
        self.w_max = w_max
        self.target_attractive_force = target_attractive_force
        self.obstacle_repulsive_weight = obstacle_repulsive_weight
        self.influence_distance = influence_distance
        self.goal_tolerance = goal_tolerance

        self.current_goal: Optional[Tuple[float, float]] = None
        self.last_resultant_force = [0.0, 0.0]
        self.last_attractive_force = [0.0, 0.0]
        self.last_repulsive_force = [0.0, 0.0]

    def set_goal(self, gx: float, gy: float):
        self.current_goal = (gx, gy)

    def is_goal_reached(self, vehicle_pose: Pose2D) -> bool:
        if self.current_goal is None:
            return True
        dist = math.hypot(self.current_goal[0] - vehicle_pose.x, self.current_goal[1] - vehicle_pose.y)
        return dist <= self.goal_tolerance

    def compute_twist(self, vehicle: VehicleBase, scanner: LaserScanner) -> Twist2D:
        """
        Computes (vx, omega) twist command using MVSim's potential field evaluation.
        """
        if self.current_goal is None:
            return Twist2D(0.0, 0.0, 0.0)

        # 1. Attractive force in world frame
        dx = self.current_goal[0] - vehicle.pose.x
        dy = self.current_goal[1] - vehicle.pose.y
        dist_to_goal = math.hypot(dx, dy)

        if dist_to_goal <= self.goal_tolerance:
            return Twist2D(0.0, 0.0, 0.0)

        # Unit vector toward goal * attractive gain
        ux = dx / dist_to_goal
        uy = dy / dist_to_goal
        f_attr_x = ux * self.target_attractive_force
        f_attr_y = uy * self.target_attractive_force

        # Convert attractive force into vehicle's local frame
        cos_yaw = math.cos(vehicle.pose.yaw)
        sin_yaw = math.sin(vehicle.pose.yaw)
        local_attr_x = f_attr_x * cos_yaw + f_attr_y * sin_yaw
        local_attr_y = -f_attr_x * sin_yaw + f_attr_y * cos_yaw
        self.last_attractive_force = [local_attr_x, local_attr_y]

        # 2. Repulsive force from LiDAR scan beams (in vehicle local frame)
        f_rep_x = 0.0
        f_rep_y = 0.0

        n_rays = len(scanner.ranges)
        if n_rays > 0:
            for idx, r in enumerate(scanner.ranges):
                if not scanner.valid[idx]:
                    continue
                if r <= 0.05 or r > self.influence_distance:
                    continue

                ang = scanner.ray_angles[idx] + scanner.sensor_offset.yaw

                # MVSim formula: mod = min(1e3, 1.0 / (r * r))
                # Add repulsive force away from hit point
                mod = min(200.0, 1.0 / (r * r))
                fx = -math.cos(ang) * mod
                fy = -math.sin(ang) * mod

                f_rep_x += fx * self.obstacle_repulsive_weight
                f_rep_y += fy * self.obstacle_repulsive_weight

        self.last_repulsive_force = [f_rep_x, f_rep_y]

        # 3. Resultant potential force
        res_x = local_attr_x + f_rep_x
        res_y = local_attr_y + f_rep_y
        self.last_resultant_force = [res_x, res_y]

        # 4. Map resultant force to twist (forward velocity vx and yaw rate omega)
        target_heading_error = math.atan2(res_y, res_x)

        # If obstacles directly in front or strong turn needed, slow down or turn in place
        force_mag = math.hypot(res_x, res_y)
        if force_mag > 1e-4:
            heading_factor = max(0.0, math.cos(target_heading_error))
            vx = self.v_max * heading_factor
            # If obstacle repulsive force pushes backward, permit stopping / reversing
            if res_x < 0.0:
                vx = max(-0.3, -0.4 * abs(math.cos(target_heading_error)))

            omega = 1.8 * target_heading_error
            omega = max(-self.w_max, min(self.w_max, omega))
        else:
            vx = 0.0
            omega = 0.0

        return Twist2D(vx=vx, vy=0.0, omega=omega)
