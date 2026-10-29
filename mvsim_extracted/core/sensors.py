"""
Sensors simulation module ported from mrpt/mvsim.
Directly corresponds to:
- modules/simulator/src/Sensors/LaserScanner.cpp
- modules/simulator/src/Sensors/IMU.cpp
- modules/simulator/src/Sensors/ImuNoiseModel.cpp
"""

import math
import random
from typing import List, Tuple, Optional, Any
from .kinematics import Pose2D, VehicleBase


class LaserScanner:
    """
    2D LiDAR Laser Scanner sensor ported directly from MVSim's LaserScanner.cpp.
    """
    def __init__(
        self,
        name: str = "laser1",
        sensor_x: float = 0.20,
        sensor_y: float = 0.0,
        sensor_yaw: float = 0.0,
        fov_degrees: float = 180.0,
        n_rays: int = 181,
        max_range: float = 12.0,
        range_std_noise: float = 0.015
    ):
        self.name = name
        self.sensor_offset = Pose2D(sensor_x, sensor_y, math.radians(sensor_yaw))
        self.aperture = math.radians(fov_degrees)
        self.n_rays = n_rays
        self.max_range = max_range
        self.range_std_noise = range_std_noise

        if self.n_rays > 1:
            self.d_angle = self.aperture / (self.n_rays - 1)
        else:
            self.d_angle = 0.0

        self.ray_angles: List[float] = [
            -0.5 * self.aperture + i * self.d_angle for i in range(self.n_rays)
        ]

        self.ranges: List[float] = [max_range] * self.n_rays
        self.valid: List[bool] = [False] * self.n_rays
        self.hit_points_world: List[Tuple[float, float]] = []
        self.hit_points_local: List[Tuple[float, float]] = []

    def get_global_sensor_pose(self, vehicle_pose: Pose2D) -> Pose2D:
        gx, gy = vehicle_pose.transform_point(self.sensor_offset.x, self.sensor_offset.y)
        gyaw = vehicle_pose.yaw + self.sensor_offset.yaw
        res = Pose2D(gx, gy, gyaw)
        res.normalize_yaw()
        return res

    def scan(self, vehicle: VehicleBase, world: Any) -> List[float]:
        global_sensor_pose = self.get_global_sensor_pose(vehicle.pose)
        sx, sy = global_sensor_pose.x, global_sensor_pose.y
        s_yaw = global_sensor_pose.yaw

        segments = world.get_all_obstacle_segments(exclude_vehicle=vehicle)

        self.ranges = [self.max_range] * self.n_rays
        self.valid = [False] * self.n_rays
        self.hit_points_world.clear()
        self.hit_points_local.clear()

        cos_table = [math.cos(s_yaw + ang) for ang in self.ray_angles]
        sin_table = [math.sin(s_yaw + ang) for ang in self.ray_angles]

        for i in range(self.n_rays):
            dx = cos_table[i]
            dy = sin_table[i]
            ex = sx + dx * self.max_range
            ey = sy + dy * self.max_range

            min_dist = self.max_range
            hit_x, hit_y = ex, ey
            hit = False

            for (x1, y1, x2, y2) in segments:
                dist, hx, hy = self._ray_segment_intersection(sx, sy, ex, ey, x1, y1, x2, y2)
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    hit_x, hit_y = hx, hy
                    hit = True

            if hit:
                if self.range_std_noise > 0.0:
                    noisy_dist = min_dist + random.gauss(0.0, self.range_std_noise)
                    min_dist = max(0.02, min(self.max_range, noisy_dist))
                self.ranges[i] = min_dist
                self.valid[i] = True
                self.hit_points_world.append((hit_x, hit_y))

                lx, ly = vehicle.pose.inverse_transform_point(hit_x, hit_y)
                self.hit_points_local.append((lx, ly))
            else:
                self.ranges[i] = self.max_range
                self.valid[i] = False

        return self.ranges

    @staticmethod
    def _ray_segment_intersection(
        rx1: float, ry1: float, rx2: float, ry2: float,
        sx1: float, sy1: float, sx2: float, sy2: float
    ) -> Tuple[Optional[float], float, float]:
        rdx = rx2 - rx1
        rdy = ry2 - ry1
        sdx = sx2 - sx1
        sdy = sy2 - sy1

        denom = rdx * sdy - rdy * sdx
        if abs(denom) < 1e-9:
            return None, 0.0, 0.0

        t = ((sx1 - rx1) * sdy - (sy1 - ry1) * sdx) / denom
        u = ((sx1 - rx1) * rdy - (sy1 - ry1) * rdx) / denom

        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            hit_x = rx1 + t * rdx
            hit_y = ry1 + t * rdy
            dist = math.hypot(hit_x - rx1, hit_y - ry1)
            return dist, hit_x, hit_y

        return None, 0.0, 0.0


class IMUSensor:
    """
    IMU sensor ported from MVSim's IMU.cpp & ImuNoiseModel.cpp.
    """
    def __init__(
        self,
        name: str = "imu1",
        accel_noise: float = 0.05,
        gyro_noise: float = 0.02,
        bias_drift: float = 0.001
    ):
        self.name = name
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.bias_drift = bias_drift

        self.gyro_bias = 0.0
        self.accel_bias_x = 0.0
        self.accel_bias_y = 0.0

        self.prev_vx = 0.0
        self.prev_vy = 0.0

        self.ax = 0.0
        self.ay = 0.0
        self.wz = 0.0

    def update(self, vehicle: VehicleBase, dt: float):
        if dt <= 0.0:
            return

        dvx = (vehicle.twist.vx - self.prev_vx) / dt
        dvy = (vehicle.twist.vy - self.prev_vy) / dt
        self.prev_vx = vehicle.twist.vx
        self.prev_vy = vehicle.twist.vy

        wz_true = vehicle.twist.omega

        self.gyro_bias += random.gauss(0.0, self.bias_drift)
        self.accel_bias_x += random.gauss(0.0, self.bias_drift)
        self.accel_bias_y += random.gauss(0.0, self.bias_drift)

        self.ax = dvx + self.accel_bias_x + random.gauss(0.0, self.accel_noise)
        self.ay = dvy + self.accel_bias_y + random.gauss(0.0, self.accel_noise)
        self.wz = wz_true + self.gyro_bias + random.gauss(0.0, self.gyro_noise)
