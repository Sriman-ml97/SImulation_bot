"""
Intelligent Autonomous Navigation Brain for MVSim.
Combines:
1. Occupancy Grid SLAM (active environment learning & discovery)
2. Global A* Path Planning (navigates through aisles, around obstacles, avoids dead-ends)
3. Dynamic Replanning on obstacle discovery
4. Pure Pursuit Waypoint Tracking with APF local safety layer
5. Automated unsticking and recovery behaviors
"""

import math
import time
from typing import Tuple, List, Optional, Dict, Any
from .kinematics import VehicleBase, Twist2D, Pose2D
from .sensors import LaserScanner
from .occupancy_grid import OccupancyGrid
from .path_planner import AStarPlanner
from .apf_controller import APFObstacleAvoidance
from .world import NamedTarget


class IntelligentNavigator:
    """
    Intelligent Navigation Controller that learns the environment, manages multiple
    named targets, notifies target arrival, and automatically advances to the next environment target.
    """
    def __init__(
        self,
        world_width: float = 34.0,
        world_height: float = 26.0,
        v_max: float = 1.35,
        w_max: float = 2.2,
        goal_tolerance: float = 0.40
    ):
        self.v_max = v_max
        self.w_max = w_max
        self.goal_tolerance = goal_tolerance

        # SLAM & Planning components
        self.occ_grid = OccupancyGrid(world_width, world_height, resolution=0.20)
        self.planner = AStarPlanner()
        self.local_apf = APFObstacleAvoidance(
            v_max=v_max,
            w_max=w_max,
            influence_distance=1.6,
            obstacle_repulsive_weight=0.9
        )

        # Multi-Target Management
        self.targets: List[NamedTarget] = []
        self.current_target_index: int = 0
        self.active_target: Optional[NamedTarget] = None
        self.completed_targets_count: int = 0
        self.target_reached_notification: Optional[Dict[str, Any]] = None
        self.dwell_until: float = 0.0

        # Navigation State
        self.current_goal: Optional[Tuple[float, float]] = None
        self.active_path: List[Tuple[float, float]] = []
        self.current_waypoint_idx: int = 0
        self.state: str = "IDLE"  # "IDLE", "NAVIGATING", "GOAL_REACHED", "RECOVERING"

        # Timing and replanning
        self.last_replan_time: float = 0.0
        self.replan_interval: float = 0.6  # Seconds between regular dynamic replans

        # Stuck detection
        self.position_history: List[Tuple[float, float, float]] = []  # (x, y, time)
        self.recovery_end_time: float = 0.0
        self.recovery_turn_dir: float = 1.0

    def set_targets(self, targets: List[NamedTarget]):
        """Sets the sequence of named environment targets for the mission tour."""
        self.targets = list(targets)
        self.current_target_index = 0
        self.completed_targets_count = 0
        self.dwell_until = 0.0
        self.target_reached_notification = None

        if self.targets:
            self.active_target = self.targets[0]
            self.set_goal(self.active_target.x, self.active_target.y)
            print(f"[CAMPUS MISSION] Initialized mission with {len(self.targets)} targets. Active Goal: '{self.active_target.name}'")
        else:
            self.active_target = None
            self.current_goal = None

    def add_custom_target(self, name: str, x: float, y: float, description: str = ""):
        """Adds a new named target dynamically to the mission queue."""
        t = NamedTarget(name, x, y, description)
        self.targets.append(t)
        if self.active_target is None or self.state == "GOAL_REACHED":
            self.active_target = t
            self.set_goal(x, y)
        print(f"[CAMPUS MISSION] Added New Target: '{name}' at ({x:.2f}, {y:.2f})")
        return t

    def skip_to_next_target(self):
        """Manually advances to the next environment target."""
        if not self.targets:
            return
        self.current_target_index = (self.current_target_index + 1) % len(self.targets)
        self.active_target = self.targets[self.current_target_index]
        self.dwell_until = 0.0
        self.set_goal(self.active_target.x, self.active_target.y)
        print(f"[CAMPUS MISSION] Switched to Destination: '{self.active_target.name}'")

    def set_goal(self, gx: float, gy: float, force_replan: bool = True):
        """Sets a new target goal and triggers global path planning."""
        self.current_goal = (gx, gy)
        self.state = "NAVIGATING"
        self.current_waypoint_idx = 0
        if force_replan:
            self.last_replan_time = 0.0  # Force immediate replan on next step

    def is_goal_reached(self, pose: Pose2D) -> bool:
        if self.current_goal is None:
            return True
        dist = math.hypot(self.current_goal[0] - pose.x, self.current_goal[1] - pose.y)
        return dist <= self.goal_tolerance

    def update_and_navigate(
        self,
        vehicle: VehicleBase,
        scanner: LaserScanner,
        dt: float
    ) -> Twist2D:
        """
        Executes one navigation cycle:
        1. Learns the environment by updating the Occupancy Grid with LiDAR data
        2. Evaluates goal distance, triggers arrival notification, and sequences next target
        3. Replans if needed
        4. Calculates optimal twist command
        """
        now = time.time()

        # 1. Update Occupancy Grid SLAM (Learning the environment)
        sensor_pose = scanner.get_global_sensor_pose(vehicle.pose)
        self.occ_grid.update_from_lidar(
            sensor_world_x=sensor_pose.x,
            sensor_world_y=sensor_pose.y,
            hit_points=scanner.hit_points_world,
            ranges=scanner.ranges,
            ray_angles=scanner.ray_angles,
            sensor_yaw=sensor_pose.yaw,
            max_range=scanner.max_range
        )

        # Check if robot is currently dwelling at a notified target
        if now < self.dwell_until:
            return Twist2D(0.0, 0.0, 0.0)
        elif self.dwell_until > 0.0 and now >= self.dwell_until:
            # Dwell elapsed -> automatically set the next environment target!
            self.dwell_until = 0.0
            if self.targets:
                self.current_target_index = (self.current_target_index + 1) % len(self.targets)
                self.active_target = self.targets[self.current_target_index]
                self.set_goal(self.active_target.x, self.active_target.y)
                self.state = "NAVIGATING"
                print(f"[CAMPUS MISSION] Departing for Next Department Target: '{self.active_target.name}'")

        if self.current_goal is None:
            self.state = "IDLE"
            return Twist2D(0.0, 0.0, 0.0)

        # 2. Check if active target goal is reached
        dist_to_goal = math.hypot(self.current_goal[0] - vehicle.pose.x, self.current_goal[1] - vehicle.pose.y)
        if dist_to_goal <= self.goal_tolerance:
            self.state = "GOAL_REACHED"
            self.active_path.clear()

            # Trigger NOTIFICATION on target arrival
            if self.active_target and not self.active_target.reached:
                self.active_target.mark_reached(now)
                self.completed_targets_count += 1
                stats = self.occ_grid.get_exploration_stats()

                notif = {
                    "name": self.active_target.name,
                    "x": self.active_target.x,
                    "y": self.active_target.y,
                    "description": self.active_target.description,
                    "timestamp": now,
                    "index": self.current_target_index + 1,
                    "total": len(self.targets),
                    "percent_explored": stats["percent_explored"]
                }
                self.target_reached_notification = notif

                # Dwell time before departing to next environment target
                dwell = self.active_target.dwell_time if self.active_target.dwell_time > 0 else 2.0
                self.dwell_until = now + dwell

                next_idx = (self.current_target_index + 1) % len(self.targets)
                next_name = self.targets[next_idx].name if self.targets else "None"

                print("\n" + "=" * 65)
                print(f"[CAMPUS NOTIFICATION] TARGET REACHED & NOTIFIED!")
                print(f"  - Destination : '{self.active_target.name}'")
                print(f"  - Coordinates : ({self.active_target.x:.2f}m, {self.active_target.y:.2f}m)")
                print(f"  - Mission     : {self.completed_targets_count} of {len(self.targets)} college targets completed")
                print(f"  - Campus Map  : {stats['percent_explored']:.1f}% explored ({stats['obstacle_cells']} obstacles mapped)")
                if self.active_target.description:
                    print(f"  - Task Details: {self.active_target.description}")
                print(f"  - Status      : Notified target! Proceeding to next destination: '{next_name}' in {dwell:.1f}s")
                print("=" * 65 + "\n")

            return Twist2D(0.0, 0.0, 0.0)

        # 3. Check for Stuck Condition
        self.position_history.append((vehicle.pose.x, vehicle.pose.y, now))
        # Keep 1.5 seconds of history
        self.position_history = [p for p in self.position_history if (now - p[2]) <= 1.5]

        if len(self.position_history) >= 10 and self.state == "NAVIGATING":
            start_p = self.position_history[0]
            curr_p = self.position_history[-1]
            disp = math.hypot(curr_p[0] - start_p[0], curr_p[1] - start_p[1])
            # If moved less than 0.08m over 1.5 seconds while navigating
            if disp < 0.08 and now > self.recovery_end_time:
                self.state = "RECOVERING"
                self.recovery_end_time = now + 0.8  # Recover for 0.8s
                # Turn away from closest obstacle
                min_idx = min(range(len(scanner.ranges)), key=lambda i: scanner.ranges[i]) if scanner.ranges else 0
                self.recovery_turn_dir = -1.0 if scanner.ray_angles[min_idx] > 0 else 1.0

        # Handle active recovery behavior
        if self.state == "RECOVERING":
            if now < self.recovery_end_time:
                # Back up while rotating away from obstruction
                return Twist2D(vx=-0.35, vy=0.0, omega=self.recovery_turn_dir * 1.8)
            else:
                self.state = "NAVIGATING"
                self.position_history.clear()
                self.last_replan_time = 0.0  # Replan immediately after unstick

        # 4. Global A* Replanning
        needs_replan = (
            not self.active_path
            or (now - self.last_replan_time) >= self.replan_interval
            or self._is_path_obstructed()
        )

        if needs_replan:
            self.last_replan_time = now
            planned = self.planner.plan(
                start_world=(vehicle.pose.x, vehicle.pose.y),
                goal_world=self.current_goal,
                occ_grid=self.occ_grid
            )
            if planned and len(planned) >= 2:
                self.active_path = planned
                self.current_waypoint_idx = 1
            elif not self.active_path:
                # Fallback directly to target if planner couldn't find discretised path
                self.active_path = [(vehicle.pose.x, vehicle.pose.y), self.current_goal]
                self.current_waypoint_idx = 1

        # 5. Advance waypoint index along smoothed path
        while self.current_waypoint_idx < len(self.active_path) - 1:
            wp = self.active_path[self.current_waypoint_idx]
            d_wp = math.hypot(wp[0] - vehicle.pose.x, wp[1] - vehicle.pose.y)
            if d_wp < 0.75:  # Within waypoint capture radius
                self.current_waypoint_idx += 1
            else:
                break

        # Select target waypoint
        if self.current_waypoint_idx < len(self.active_path):
            target_wp = self.active_path[self.current_waypoint_idx]
        else:
            target_wp = self.current_goal

        # 6. Waypoint Tracking & Heading Control (Pure Pursuit + Local APF)
        dx = target_wp[0] - vehicle.pose.x
        dy = target_wp[1] - vehicle.pose.y
        dist_to_wp = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - vehicle.pose.yaw
        # Normalize angle to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2.0 * math.pi) - math.pi

        # Speed scaling based on alignment
        align_factor = max(0.0, math.cos(heading_error))
        # Slow down near final goal
        speed_factor = min(1.0, max(0.25, dist_to_goal / 1.5))
        vx = self.v_max * align_factor * speed_factor

        # Sharper turns when misaligned
        omega = 2.4 * heading_error
        omega = max(-self.w_max, min(self.w_max, omega))

        # Local reactive collision buffer from immediate LiDAR returns
        # If any obstacle is closer than 0.45m in the forward arc, apply repulsive steering
        for idx in range(len(scanner.ranges)):
            if not scanner.valid[idx]:
                continue
            r = scanner.ranges[idx]
            if r < 0.50:
                ang = scanner.ray_angles[idx]
                if abs(ang) < math.radians(65):  # In front arc
                    # Steer away from obstacle side
                    rep_steer = -math.copysign(1.6 / max(0.15, r), ang)
                    omega += rep_steer * dt * 25.0
                    omega = max(-self.w_max, min(self.w_max, omega))
                    vx *= 0.5  # Slow down when near obstacle
                    break

        return Twist2D(vx=vx, vy=0.0, omega=omega)

    def _is_path_obstructed(self) -> bool:
        """Checks if upcoming path waypoints are blocked by newly discovered obstacles."""
        if not self.active_path or self.current_waypoint_idx >= len(self.active_path):
            return False

        costmap = self.occ_grid.get_inflated_costmap()
        # Check next 2 waypoints
        end_idx = min(len(self.active_path), self.current_waypoint_idx + 2)
        for i in range(self.current_waypoint_idx, end_idx):
            col, row = self.occ_grid.world_to_grid(*self.active_path[i])
            if self.occ_grid.is_in_bounds(col, row):
                if costmap[row, col] >= 200:
                    return True
        return False
