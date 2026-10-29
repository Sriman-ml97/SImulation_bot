"""
Interactive visualizer and simulation runner using OpenCV.
Renders real-time vehicle dynamics, LiDAR point clouds, APF navigation vectors,
and provides live teleoperation matching MVSim's GUI panels.
"""

import math
import time
import cv2
import numpy as np
from typing import Optional, List, Tuple
try:
    from .core.world import World
    from .core.kinematics import DifferentialVehicle, AckermannVehicle
    from .core.sensors import LaserScanner, IMUSensor
    from .core.apf_controller import APFObstacleAvoidance
    from .core.intelligent_navigator import IntelligentNavigator
except (ImportError, ValueError):
    from core.world import World
    from core.kinematics import DifferentialVehicle, AckermannVehicle
    from core.sensors import LaserScanner, IMUSensor
    from core.apf_controller import APFObstacleAvoidance
    from core.intelligent_navigator import IntelligentNavigator


class MVSimVisualizer:
    def __init__(
        self,
        world: World,
        canvas_width: int = 1000,
        canvas_height: int = 750,
        ppm: float = 32.0  # pixels per meter
    ):
        self.world = world
        self.width = canvas_width
        self.height = canvas_height
        self.ppm = ppm

        # Center of canvas corresponds to world (0, 0)
        self.origin_x = canvas_width // 2
        self.origin_y = canvas_height // 2

        # Active vehicle and sensors
        self.active_vehicle_name = next(iter(world.vehicles.keys())) if world.vehicles else None
        self.scanners: dict[str, LaserScanner] = {}
        self.imus: dict[str, IMUSensor] = {}
        self.apf_controllers: dict[str, APFObstacleAvoidance] = {}
        self.navigators: dict[str, IntelligentNavigator] = {}

        self.current_goal: Tuple[float, float] = (0.0, -11.0)
        if world.targets:
            self.current_goal = (world.targets[0].x, world.targets[0].y)

        for name, veh in world.vehicles.items():
            self.scanners[name] = LaserScanner(name=f"{name}_lidar", n_rays=181, fov_degrees=270.0, max_range=14.0)
            self.imus[name] = IMUSensor(name=f"{name}_imu")
            self.apf_controllers[name] = APFObstacleAvoidance(v_max=1.35, w_max=2.2)
            self.navigators[name] = IntelligentNavigator(
                world_width=world.width,
                world_height=world.height,
                v_max=1.35,
                w_max=2.2
            )
            if world.targets:
                self.navigators[name].set_targets(world.targets)
            else:
                self.navigators[name].set_goal(*self.current_goal)

        # Simulation state
        self.autonomous_mode = False
        self.show_lidar_rays = True
        self.show_telemetry = True
        self.show_learned_map = True
        self.trajectory_history: dict[str, List[Tuple[int, int]]] = {name: [] for name in world.vehicles}

    def set_target_goal(self, gx: float, gy: float):
        """Sets a new goal target for visualizer and all navigators."""
        self.current_goal = (gx, gy)
        for nav in self.navigators.values():
            nav.set_goal(gx, gy)

    def skip_to_next_target(self):
        """Manually advances all robot navigators to the next college target."""
        for nav in self.navigators.values():
            nav.skip_to_next_target()
        active_nav = self.navigators.get(self.active_vehicle_name)
        if active_nav and active_nav.active_target:
            self.current_goal = (active_nav.active_target.x, active_nav.active_target.y)

    def add_custom_target(self, name: str, wx: float, wy: float):
        """Adds a new custom target to the environment and active navigators."""
        t = self.world.add_target(name, wx, wy)
        for nav in self.navigators.values():
            nav.add_custom_target(name, wx, wy)
        self.current_goal = (wx, wy)

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int(self.origin_x + wx * self.ppm)
        sy = int(self.origin_y - wy * self.ppm)  # flip y for screen coordinates
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        wx = (sx - self.origin_x) / self.ppm
        wy = (self.origin_y - sy) / self.ppm
        return wx, wy

    def step_and_render(self, dt: float, keys_pressed: dict) -> np.ndarray:
        # Create dark background matching MVSim cinematic dark aesthetic (#0a0c12)
        frame = np.full((self.height, self.width, 3), (18, 12, 10), dtype=np.uint8)

        # 1. Update sensors for all vehicles first
        for name, veh in self.world.vehicles.items():
            scanner = self.scanners[name]
            scanner.scan(veh, self.world)
            self.imus[name].update(veh, dt)

        # 2. Process controls & navigation for active vehicle
        active_veh = self.world.vehicles.get(self.active_vehicle_name)
        active_scanner = self.scanners.get(self.active_vehicle_name) if active_veh else None
        active_nav = self.navigators.get(self.active_vehicle_name) if active_veh else None

        if active_veh and active_scanner and active_nav:
            if self.autonomous_mode:
                twist = active_nav.update_and_navigate(active_veh, active_scanner, dt)
                if isinstance(active_veh, AckermannVehicle):
                    steer = math.atan2(twist.omega * active_veh.wheelbase, max(0.2, abs(twist.vx)))
                    active_veh.set_steer_and_velocity(steer, twist.vx)
                else:
                    active_veh.set_target_twist(twist.vx, twist.vy, twist.omega)
            else:
                # In manual teleop, the robot still learns the environment from LiDAR
                sensor_pose = active_scanner.get_global_sensor_pose(active_veh.pose)
                active_nav.occ_grid.update_from_lidar(
                    sensor_world_x=sensor_pose.x,
                    sensor_world_y=sensor_pose.y,
                    hit_points=active_scanner.hit_points_world,
                    ranges=active_scanner.ranges,
                    ray_angles=active_scanner.ray_angles,
                    sensor_yaw=sensor_pose.yaw,
                    max_range=active_scanner.max_range
                )
                self._handle_teleop_input(active_veh, keys_pressed)

        # 3. Step physics simulation
        self.world.step(dt)

        # 4. Record trajectory points
        for name, veh in self.world.vehicles.items():
            sp = self.world_to_screen(veh.pose.x, veh.pose.y)
            hist = self.trajectory_history[name]
            if not hist or math.hypot(hist[-1][0] - sp[0], hist[-1][1] - sp[1]) > 4:
                hist.append(sp)
                if len(hist) > 300:
                    hist.pop(0)

        # 5. Render Learned Environment (SLAM Occupancy Map)
        if active_nav and self.show_learned_map:
            self._draw_learned_map(frame, active_nav)

        # 6. Render World elements (grid, obstacles)
        self._draw_grid(frame)
        self._draw_obstacles(frame)

        # 7. Render Trajectories
        self._draw_trajectories(frame)

        # 8. Render Active LiDAR beams and hit points
        if active_veh and active_scanner and self.show_lidar_rays:
            self._draw_lidar(frame, active_veh, active_scanner)

        # 9. Render Goal and Planned A* Path
        if active_veh and active_nav:
            self._draw_goal_and_planned_path(frame, active_veh, active_nav)

        # 10. Render Vehicles
        self._draw_vehicles(frame)

        # 11. Render HUD & Telemetry
        if self.show_telemetry and active_veh and active_nav:
            self._draw_hud(frame, active_veh, active_nav, dt)

        return frame

    def _handle_teleop_input(self, veh, keys):
        target_vx = 0.0
        target_w = 0.0
        v_step = 1.4
        w_step = 2.0

        if keys.get(ord('w')) or keys.get(ord('W')):
            target_vx += v_step
        if keys.get(ord('s')) or keys.get(ord('S')):
            target_vx -= v_step * 0.7
        if keys.get(ord('a')) or keys.get(ord('A')):
            target_w += w_step
        if keys.get(ord('d')) or keys.get(ord('D')):
            target_w -= w_step
        if keys.get(32):  # Spacebar brake
            target_vx = 0.0
            target_w = 0.0

        if isinstance(veh, AckermannVehicle):
            steer = 0.0
            if keys.get(ord('a')) or keys.get(ord('A')):
                steer += veh.max_steer
            if keys.get(ord('d')) or keys.get(ord('D')):
                steer -= veh.max_steer
            veh.set_steer_and_velocity(steer, target_vx)
        else:
            veh.set_target_twist(target_vx, 0.0, target_w)

    def _draw_grid(self, frame):
        # 1-meter grid lines
        grid_color = (30, 24, 20)
        axis_color = (55, 45, 35)

        hw = int(self.world.width / 2.0)
        hh = int(self.world.height / 2.0)

        for x in range(-hw, hw + 1):
            p1 = self.world_to_screen(x, -hh)
            p2 = self.world_to_screen(x, hh)
            col = axis_color if x == 0 else grid_color
            cv2.line(frame, p1, p2, col, 1)

        for y in range(-hh, hh + 1):
            p1 = self.world_to_screen(-hw, y)
            p2 = self.world_to_screen(hw, y)
            col = axis_color if y == 0 else grid_color
            cv2.line(frame, p1, p2, col, 1)

    def _draw_obstacles(self, frame):
        # Draw bounding walls
        for wall in self.world.walls:
            p1 = self.world_to_screen(wall.x1, wall.y1)
            p2 = self.world_to_screen(wall.x2, wall.y2)
            cv2.line(frame, p1, p2, (80, 80, 95), 3, cv2.LINE_AA)

        # Draw boxes and department building labels
        for box in self.world.boxes:
            corners = box.get_corners()
            pts = np.array([self.world_to_screen(cx, cy) for cx, cy in corners], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Filled polygon with department architectural color
            fill_col = box.color
            border_col = (min(255, int(box.color[0] * 1.35)), min(255, int(box.color[1] * 1.35)), min(255, int(box.color[2] * 1.35)))
            cv2.fillPoly(frame, [pts], fill_col)
            cv2.polylines(frame, [pts], True, border_col, 2, cv2.LINE_AA)

            # Draw department/building text label on college buildings
            if hasattr(box, 'label') and box.label:
                cp = self.world_to_screen(box.cx, box.cy)
                font_scale = 0.40
                (tw, th), _ = cv2.getTextSize(box.label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
                cv2.putText(frame, box.label, (cp[0] - tw // 2, cp[1] + th // 2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_trajectories(self, frame):
        for name, hist in self.trajectory_history.items():
            if len(hist) > 1:
                col = (0, 180, 240) if name == self.active_vehicle_name else (120, 120, 120)
                for i in range(len(hist) - 1):
                    alpha = (i + 1) / len(hist)
                    c = (int(col[0] * alpha), int(col[1] * alpha), int(col[2] * alpha))
                    cv2.line(frame, hist[i], hist[i + 1], c, 1, cv2.LINE_AA)

    def _draw_lidar(self, frame, veh, scanner):
        g_pose = scanner.get_global_sensor_pose(veh.pose)
        sp_sensor = self.world_to_screen(g_pose.x, g_pose.y)

        # Draw rays with low opacity
        for i in range(0, scanner.n_rays, 2):
            r = scanner.ranges[i]
            if r >= scanner.max_range:
                continue
            ang = g_pose.yaw + scanner.ray_angles[i]
            hx = g_pose.x + r * math.cos(ang)
            hy = g_pose.y + r * math.sin(ang)
            sp_hit = self.world_to_screen(hx, hy)
            cv2.line(frame, sp_sensor, sp_hit, (50, 45, 20), 1, cv2.LINE_AA)

        # Draw hit points (glowing cyan / neon green dots)
        for hx, hy in scanner.hit_points_world:
            sp = self.world_to_screen(hx, hy)
            cv2.circle(frame, sp, 2, (255, 220, 0), -1, cv2.LINE_AA)

    def _draw_learned_map(self, frame: np.ndarray, nav: IntelligentNavigator):
        """
        Renders the discovered environment map:
        - Subtle teal/cyan tint for explored free space
        - Glowing amber markers for discovered obstacles
        """
        grid = nav.occ_grid.grid

        # 1. Subtle tint for explored free space cells (downsampled for high FPS)
        free_rows, free_cols = np.where(grid == nav.occ_grid.CELL_FREE)
        if len(free_rows) > 0:
            step = max(1, len(free_rows) // 600)
            for r, c in zip(free_rows[::step], free_cols[::step]):
                wx, wy = nav.occ_grid.grid_to_world(c, r)
                sx, sy = self.world_to_screen(wx, wy)
                cv2.circle(frame, (sx, sy), 2, (38, 30, 16), -1)

        # 2. Glowing markers for discovered obstacles
        obs_rows, obs_cols = np.where(grid == nav.occ_grid.CELL_OBSTACLE)
        if len(obs_rows) > 0:
            step = max(1, len(obs_rows) // 400)
            for r, c in zip(obs_rows[::step], obs_cols[::step]):
                wx, wy = nav.occ_grid.grid_to_world(c, r)
                sx, sy = self.world_to_screen(wx, wy)
                cv2.circle(frame, (sx, sy), 2, (0, 165, 255), -1, cv2.LINE_AA)

    def _draw_goal_and_planned_path(self, frame: np.ndarray, veh, nav: IntelligentNavigator):
        now = time.time()

        # 1. Render A* planned path
        if nav.active_path and len(nav.active_path) >= 2:
            pts = [self.world_to_screen(wx, wy) for (wx, wy) in nav.active_path]
            for i in range(len(pts) - 1):
                # Glowing neon emerald trajectory line
                cv2.line(frame, pts[i], pts[i + 1], (0, 240, 140), 2, cv2.LINE_AA)
            for idx, pt in enumerate(pts):
                is_curr = (idx == nav.current_waypoint_idx)
                col = (0, 255, 255) if is_curr else (0, 200, 90)
                radius = 5 if is_curr else 3
                cv2.circle(frame, pt, radius, col, -1, cv2.LINE_AA)

            # Line from vehicle to current target waypoint
            if nav.current_waypoint_idx < len(nav.active_path):
                tgt_pt = pts[nav.current_waypoint_idx]
                veh_pt = self.world_to_screen(veh.pose.x, veh.pose.y)
                cv2.line(frame, veh_pt, tgt_pt, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Render All Targets in the College Campus Mission
        target_list = nav.targets if nav.targets else self.world.targets
        for target in target_list:
            tp = self.world_to_screen(target.x, target.y)
            is_active = (target is nav.active_target) or (abs(target.x - self.current_goal[0]) < 0.2 and abs(target.y - self.current_goal[1]) < 0.2)
            is_done = target.reached

            if is_done:
                # Green checkmark circle for completed targets
                cv2.circle(frame, tp, 9, (0, 200, 90), 2, cv2.LINE_AA)
                cv2.circle(frame, tp, 4, (0, 200, 90), -1, cv2.LINE_AA)
                cv2.putText(frame, f"[DONE] {target.name}", (tp[0] + 13, tp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 90), 1, cv2.LINE_AA)
            elif is_active:
                # Glowing pulsing beacon for active destination
                is_dwelling = (nav.dwell_until > now)
                active_color = (0, 255, 255) if is_dwelling else (0, 240, 120)
                cv2.circle(frame, tp, 14, active_color, 2, cv2.LINE_AA)
                cv2.circle(frame, tp, 6, active_color, -1, cv2.LINE_AA)
                status_suffix = "[ARRIVED & NOTIFIED]" if is_dwelling else "[ACTIVE GOAL]"
                cv2.putText(frame, f"{target.name} {status_suffix}", (tp[0] + 16, tp[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 0.42, active_color, 1, cv2.LINE_AA)
            else:
                # Pending future targets
                cv2.circle(frame, tp, 7, (150, 130, 80), 1, cv2.LINE_AA)
                cv2.putText(frame, target.name, (tp[0] + 11, tp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

        # 3. Floating Target Arrival Notification Toast Banner
        if nav.target_reached_notification:
            notif = nav.target_reached_notification
            elapsed = now - notif["timestamp"]
            if elapsed < 4.5:
                banner_w, banner_h = 620, 50
                bx = (self.width - banner_w) // 2
                by = 16

                # Glowing card border and dark glass background
                cv2.rectangle(frame, (bx - 2, by - 2), (bx + banner_w + 2, by + banner_h + 2), (0, 240, 130), 2)
                cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (14, 22, 16), -1)

                title_txt = f"★ [NOTIFIED TARGET] ARRIVED AT: {notif['name']} ({notif['index']}/{notif['total']}) ★"
                cv2.putText(frame, title_txt, (bx + 16, by + 21), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA)

                desc_txt = f"{notif['description']} | Campus Explored: {notif['percent_explored']:.1f}%"
                cv2.putText(frame, desc_txt, (bx + 16, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 235, 215), 1, cv2.LINE_AA)

    def _draw_vehicles(self, frame):
        for name, veh in self.world.vehicles.items():
            is_active = (name == self.active_vehicle_name)
            poly = veh.get_world_polygon()
            pts = np.array([self.world_to_screen(px, py) for px, py in poly], np.int32).reshape((-1, 1, 2))

            # Chassis fill & stroke
            fill_col = (40, 160, 220) if is_active else (80, 80, 90)
            stroke_col = (80, 220, 255) if is_active else (140, 140, 150)
            cv2.fillPoly(frame, [pts], fill_col)
            cv2.polylines(frame, [pts], True, stroke_col, 2, cv2.LINE_AA)

            # Center position and heading indicator
            vp = self.world_to_screen(veh.pose.x, veh.pose.y)
            heading_len = 0.6
            hx = veh.pose.x + heading_len * math.cos(veh.pose.yaw)
            hy = veh.pose.y + heading_len * math.sin(veh.pose.yaw)
            hp = self.world_to_screen(hx, hy)
            cv2.arrowedLine(frame, vp, hp, (0, 255, 255), 2, tipLength=0.3)

            # Name tag
            cv2.putText(frame, veh.name, (vp[0] - 15, vp[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    def _draw_hud(self, frame: np.ndarray, veh, nav: IntelligentNavigator, dt: float):
        # Dark HUD panel in top-left
        cv2.rectangle(frame, (10, 10), (385, 270), (12, 10, 8), -1)
        cv2.rectangle(frame, (10, 10), (385, 270), (45, 40, 35), 1)

        cv2.putText(frame, "MVSIM CAMPUS NAVIGATION ENGINE", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 220, 255), 1)

        sim_time = self.world.simul_time
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(frame, f"Sim Time: {sim_time:.2f}s | FPS: {fps:.0f}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        if veh:
            cv2.putText(frame, f"Vehicle: {veh.name} [{type(veh).__name__}]", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(frame, f"Pose: ({veh.pose.x:.2f}, {veh.pose.y:.2f}) Yaw: {math.degrees(veh.pose.yaw):.1f}deg", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
            cv2.putText(frame, f"Twist: Vx={veh.twist.vx:.2f}m/s, W={veh.twist.omega:.2f}rad/s", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

            # Odometry drift comparison
            dx = veh.odometry_pose.x - veh.pose.x
            dy = veh.odometry_pose.y - veh.pose.y
            cv2.putText(frame, f"Odom Drift: dx={dx:+.2f}m, dy={dy:+.2f}m", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 220), 1)

            # Lidar info
            scanner = self.scanners.get(veh.name)
            if scanner:
                valid_hits = sum(1 for v in scanner.valid if v)
                min_range = min(scanner.ranges) if scanner.ranges else 0.0
                cv2.putText(frame, f"LiDAR: {valid_hits}/{scanner.n_rays} hits | MinDist: {min_range:.2f}m", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 180), 1)

        # SLAM and Path Planning Telemetry
        stats = nav.occ_grid.get_exploration_stats()
        cv2.putText(frame, f"SLAM Explored: {stats['percent_explored']:.1f}% ({stats['obstacle_cells']} obstacles mapped)", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 200), 1)

        # College Campus Mission Progress
        active_target_name = nav.active_target.name if nav.active_target else "Manual Goal"
        total_targets = len(nav.targets) if nav.targets else len(self.world.targets)
        cv2.putText(frame, f"Mission: {nav.completed_targets_count}/{total_targets} Targets Achieved", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1)

        dist_to_goal = math.hypot(self.current_goal[0] - veh.pose.x, self.current_goal[1] - veh.pose.y)
        path_wps = len(nav.active_path)
        cv2.putText(frame, f"Destination: {active_target_name} ({dist_to_goal:.1f}m)", (20, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 120), 1)

        is_dwelling = (nav.dwell_until > time.time())
        state_str = "NOTIFIED TARGET! DWELLING..." if is_dwelling else nav.state
        state_col = (0, 255, 255) if is_dwelling else ((0, 255, 120) if nav.state == "NAVIGATING" else (0, 165, 255))
        cv2.putText(frame, f"Status: {state_str} | Waypoints: {path_wps}", (20, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.40, state_col, 1)

        mode_str = "AUTONOMOUS CAMPUS TOUR (SLAM + A*)" if self.autonomous_mode else "MANUAL TELEOPERATION (WASD)"
        mode_col = (0, 255, 120) if self.autonomous_mode else (255, 180, 0)
        cv2.putText(frame, f"Mode: {mode_str}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_col, 1)

        # Controls reference at bottom
        ctrl_text = "[Click Map] Add Target | [N] Next Target | [P] Toggle Auto | [G] Random | [WASD] Drive | [Q] Quit"
        cv2.putText(frame, ctrl_text, (20, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
