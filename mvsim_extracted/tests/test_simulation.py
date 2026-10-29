"""
Unit test suite verifying MVSim extracted physics, sensors, APF, and XML loader.
"""

import math
import unittest
import sys
import os

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.kinematics import Pose2D, Twist2D, DifferentialVehicle, AckermannVehicle, PIDController
from core.sensors import LaserScanner, IMUSensor
from core.world import World, ObstacleBox, ObstacleWall
from core.apf_controller import APFObstacleAvoidance
from core.xml_loader import MVSimXMLLoader


class TestMVSimExtracted(unittest.TestCase):
    def test_pose_transform(self):
        pose = Pose2D(10.0, 5.0, math.pi / 2.0)  # Facing North
        # Point (1, 0) in local robot frame (1m forward) should be (10, 6) in world frame
        gx, gy = pose.transform_point(1.0, 0.0)
        self.assertAlmostEqual(gx, 10.0, places=4)
        self.assertAlmostEqual(gy, 6.0, places=4)

        # Inverse transform
        lx, ly = pose.inverse_transform_point(10.0, 6.0)
        self.assertAlmostEqual(lx, 1.0, places=4)
        self.assertAlmostEqual(ly, 0.0, places=4)

    def test_differential_drive_kinematics(self):
        veh = DifferentialVehicle(wheel_separation=0.5, wheel_diameter=0.2)
        veh.set_target_twist(vx=1.0, vy=0.0, omega=0.0)
        # Step forward several timesteps to let acceleration ramp take effect
        for _ in range(5):
            veh.step(dt=0.1)

        # Vehicle should have moved forward along X axis
        self.assertGreater(veh.pose.x, 0.1)
        self.assertAlmostEqual(veh.pose.y, 0.0, places=3)
        # Both wheels should rotate equally for pure translation
        self.assertAlmostEqual(veh.w_left, veh.w_right, places=2)

    def test_ackermann_kinematics(self):
        car = AckermannVehicle(wheelbase=1.5, track_width=0.8)
        # Steer 20 degrees, forward speed 2.0 m/s
        car.set_steer_and_velocity(target_steer_rad=math.radians(20.0), target_v=2.0)
        car.step(dt=0.1)

        # Yaw rate should be non-zero due to front wheel steering
        self.assertGreater(car.steer_angle, 0.0)
        self.assertGreater(car.twist.omega, 0.0)

    def test_lidar_raycasting(self):
        world = World(width=20.0, height=20.0)
        # Add box directly in front at (3.0, 0.0)
        world.add_box(cx=3.0, cy=0.0, width=1.0, height=2.0)

        robot = DifferentialVehicle(init_pose=Pose2D(0.0, 0.0, 0.0))
        world.add_vehicle(robot)

        scanner = LaserScanner(n_rays=181, fov_degrees=180.0, max_range=10.0, range_std_noise=0.0)
        ranges = scanner.scan(robot, world)

        # Center ray (index 90 for 181 rays) points directly forward along X axis
        center_idx = 90
        # Distance to front face of box: 3.0 - (1.0/2) = 2.5m, plus sensor offset 0.2m = 2.3m
        center_range = ranges[center_idx]
        self.assertAlmostEqual(center_range, 2.3, delta=0.1)
        self.assertTrue(scanner.valid[center_idx])

    def test_imu_sensor(self):
        robot = DifferentialVehicle()
        imu = IMUSensor(accel_noise=0.0, gyro_noise=0.0, bias_drift=0.0)

        # Simulate acceleration
        robot.twist.vx = 2.0
        robot.twist.omega = 1.0
        imu.update(robot, dt=0.1)

        # Should register positive linear acceleration and gyro rate
        self.assertGreater(imu.ax, 0.0)
        self.assertAlmostEqual(imu.wz, 1.0, delta=0.05)

    def test_apf_obstacle_avoidance(self):
        apf = APFObstacleAvoidance(v_max=1.0, w_max=2.0)
        apf.set_goal(10.0, 0.0)

        robot = DifferentialVehicle(init_pose=Pose2D(0.0, 0.0, 0.0))
        scanner = LaserScanner(n_rays=181, max_range=10.0)

        # Open space: resultant force should point straight toward goal
        twist = apf.compute_twist(robot, scanner)
        self.assertGreater(twist.vx, 0.5)
        self.assertAlmostEqual(twist.omega, 0.0, places=1)

    def test_xml_export_and_import(self):
        w1 = World.create_preset("warehouse")
        xml_str = MVSimXMLLoader.export_to_xml_string(w1)
        self.assertIn("<mvsim_world", xml_str)
        self.assertIn("<block", xml_str)
        self.assertIn("<vehicle", xml_str)

    def test_occupancy_grid_slam(self):
        from core.occupancy_grid import OccupancyGrid
        grid = OccupancyGrid(world_width=20.0, world_height=20.0, resolution=0.2)

        # Simulate a LiDAR scan hitting an obstacle at (3.0, 0.0) from sensor at (0.0, 0.0)
        grid.update_from_lidar(
            sensor_world_x=0.0,
            sensor_world_y=0.0,
            hit_points=[(3.0, 0.0)],
            ranges=[3.0],
            ray_angles=[0.0],
            sensor_yaw=0.0,
            max_range=10.0
        )

        # Cell at (1.5, 0.0) should be free
        fc_col, fc_row = grid.world_to_grid(1.5, 0.0)
        self.assertEqual(grid.grid[fc_row, fc_col], OccupancyGrid.CELL_FREE)

        # Cell at (3.0, 0.0) should be obstacle
        oc_col, oc_row = grid.world_to_grid(3.0, 0.0)
        self.assertEqual(grid.grid[oc_row, oc_col], OccupancyGrid.CELL_OBSTACLE)

        # Exploration stats should show progress
        stats = grid.get_exploration_stats()
        self.assertGreater(stats["explored_cells"], 0)
        self.assertGreater(stats["obstacle_cells"], 0)

    def test_astar_planner_detours_around_obstacle(self):
        from core.occupancy_grid import OccupancyGrid
        from core.path_planner import AStarPlanner

        grid = OccupancyGrid(world_width=20.0, world_height=20.0, resolution=0.2)
        # Put a vertical wall blocking straight path between (-3, 0) and (3, 0)
        # Wall from y = -2 to y = 2 at x = 0
        for y_step in range(-10, 11):
            y_w = y_step * 0.2
            c, r = grid.world_to_grid(0.0, y_w)
            grid.grid[r, c] = OccupancyGrid.CELL_OBSTACLE

        planner = AStarPlanner()
        path = planner.plan(start_world=(-3.0, 0.0), goal_world=(3.0, 0.0), occ_grid=grid)

        # Path must exist and must detour around the obstacle wall
        self.assertIsNotNone(path)
        self.assertGreaterEqual(len(path), 2)
        # Intermediate waypoints must circumvent x=0 without passing through wall
        has_detour = any(abs(wp[1]) >= 1.8 for wp in path)
        self.assertTrue(has_detour)

    def test_intelligent_navigator(self):
        from core.intelligent_navigator import IntelligentNavigator
        nav = IntelligentNavigator(world_width=20.0, world_height=20.0)
        nav.set_goal(5.0, 0.0)

        robot = DifferentialVehicle(init_pose=Pose2D(0.0, 0.0, 0.0))
        world = World(20.0, 20.0)
        scanner = LaserScanner()
        scanner.scan(robot, world)

        # One step of navigation
        twist = nav.update_and_navigate(robot, scanner, dt=0.033)
    def test_college_preset_and_target_notification(self):
        from core.intelligent_navigator import IntelligentNavigator
        world = World.create_preset("college")
        self.assertEqual(len(world.targets), 7)
        self.assertIn("1. Central Library", [t.name for t in world.targets])
        self.assertIn("3. Computer Science Dept", [t.name for t in world.targets])

        nav = IntelligentNavigator(world.width, world.height, goal_tolerance=0.40)
        nav.set_targets(world.targets)

        robot = next(iter(world.vehicles.values()))
        scanner = LaserScanner()
        scanner.scan(robot, world)

        # Teleport robot close to target 1 (Library at (-9.5, 1.2))
        robot.pose.x = -9.5
        robot.pose.y = 1.25

        # Execute navigation step
        twist = nav.update_and_navigate(robot, scanner, dt=0.033)

        # Target should be reached & notified!
        self.assertTrue(nav.targets[0].reached)
        self.assertIsNotNone(nav.target_reached_notification)
        self.assertEqual(nav.target_reached_notification["name"], "1. Central Library")
        self.assertEqual(nav.completed_targets_count, 1)

        # While dwelling, twist should be 0 (stopped at department station)
        self.assertEqual(twist.vx, 0.0)
        self.assertEqual(twist.omega, 0.0)


if __name__ == "__main__":
    unittest.main()
