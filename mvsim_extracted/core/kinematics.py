"""
Core kinematics and vehicle dynamics ported and extracted from mrpt/mvsim.
Directly corresponds to:
- modules/simulator/src/VehicleDynamics/VehicleDifferential.cpp
- modules/simulator/src/VehicleDynamics/VehicleAckermann.cpp
- modules/simulator/src/VehicleDynamics/VehicleDifferential_ControllerTwistIdeal.cpp
- modules/simulator/src/PID_Controller.cpp
"""

import math
import random
from typing import Tuple, List, Optional


class Pose2D:
    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)  # in radians

    def copy(self) -> 'Pose2D':
        return Pose2D(self.x, self.y, self.yaw)

    def normalize_yaw(self):
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

    def transform_point(self, local_x: float, local_y: float) -> Tuple[float, float]:
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        gx = self.x + local_x * cos_y - local_y * sin_y
        gy = self.y + local_x * sin_y + local_y * cos_y
        return gx, gy

    def inverse_transform_point(self, gx: float, gy: float) -> Tuple[float, float]:
        dx = gx - self.x
        dy = gy - self.y
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        lx = dx * cos_y + dy * sin_y
        ly = -dx * sin_y + dy * cos_y
        return lx, ly

    def __repr__(self):
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, yaw={math.degrees(self.yaw):.1f}°)"


class Twist2D:
    def __init__(self, vx: float = 0.0, vy: float = 0.0, omega: float = 0.0):
        self.vx = float(vx)      # forward velocity [m/s]
        self.vy = float(vy)      # lateral velocity [m/s]
        self.omega = float(omega)  # angular velocity [rad/s]

    def copy(self) -> 'Twist2D':
        return Twist2D(self.vx, self.vy, self.omega)

    def __repr__(self):
        return f"Twist2D(vx={self.vx:.2f}, vy={self.vy:.2f}, omega={self.omega:.2f})"


class PIDController:
    """PID controller matching mvsim's PID_Controller.cpp"""
    def __init__(self, kp: float, ki: float, kd: float, out_min: float = -1.0, out_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target: float, actual: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        error = target - actual
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(self.out_min, min(self.out_max, output))


class VehicleBase:
    """Base class for all simulated vehicles in mvsim."""
    def __init__(self, name: str, init_pose: Optional[Pose2D] = None, mass: float = 20.0):
        self.name = name
        self.pose = init_pose.copy() if init_pose else Pose2D()
        self.twist = Twist2D()
        self.target_twist = Twist2D()
        self.mass = mass
        self.chassis_poly: List[Tuple[float, float]] = []
        self.collision_radius: float = 0.5
        self.odometry_pose = self.pose.copy()
        self.odom_noise_multiplier = (1.002, 1.001, 1.003)

    def set_target_twist(self, vx: float, vy: float, omega: float):
        self.target_twist.vx = vx
        self.target_twist.vy = vy
        self.target_twist.omega = omega

    def get_world_polygon(self) -> List[Tuple[float, float]]:
        if not self.chassis_poly:
            pts = []
            for i in range(8):
                ang = i * (2.0 * math.pi / 8.0)
                pts.append(self.pose.transform_point(
                    self.collision_radius * math.cos(ang),
                    self.collision_radius * math.sin(ang)
                ))
            return pts
        return [self.pose.transform_point(px, py) for px, py in self.chassis_poly]

    def step(self, dt: float):
        raise NotImplementedError


class DifferentialVehicle(VehicleBase):
    """
    Differential drive vehicle matching MVSim's VehicleDifferential & small_robot.vehicle.xml.
    """
    def __init__(
        self,
        name: str = "r1",
        init_pose: Optional[Pose2D] = None,
        wheel_separation: float = 0.50,
        wheel_diameter: float = 0.20,
        max_v: float = 1.8,
        max_w: float = 2.5,
        accel_limit: float = 2.5,
        alpha_limit: float = 4.0
    ):
        super().__init__(name, init_pose, mass=15.0)
        self.wheel_separation = wheel_separation
        self.wheel_radius = wheel_diameter / 2.0
        self.max_v = max_v
        self.max_w = max_w
        self.accel_limit = accel_limit
        self.alpha_limit = alpha_limit

        self.w_left: float = 0.0
        self.w_right: float = 0.0

        hl = 0.35
        hw = 0.25
        self.collision_radius = math.hypot(hl, hw)
        self.chassis_poly = [
            (-hl * 0.8, -hw),
            (-hl * 0.8, hw),
            (hl * 0.7, hw),
            (hl, hw * 0.4),
            (hl, -hw * 0.4),
            (hl * 0.7, -hw)
        ]

    def step(self, dt: float):
        if dt <= 0.0:
            return

        dv = self.target_twist.vx - self.twist.vx
        max_dv = self.accel_limit * dt
        if abs(dv) > max_dv:
            dv = math.copysign(max_dv, dv)
        self.twist.vx += dv
        self.twist.vx = max(-self.max_v, min(self.max_v, self.twist.vx))

        dw = self.target_twist.omega - self.twist.omega
        max_dw = self.alpha_limit * dt
        if abs(dw) > max_dw:
            dw = math.copysign(max_dw, dw)
        self.twist.omega += dw
        self.twist.omega = max(-self.max_w, min(self.max_w, self.twist.omega))

        self.w_right = (self.twist.vx + (self.twist.omega * self.wheel_separation / 2.0)) / self.wheel_radius
        self.w_left = (self.twist.vx - (self.twist.omega * self.wheel_separation / 2.0)) / self.wheel_radius

        theta_mid = self.pose.yaw + 0.5 * self.twist.omega * dt
        dx = self.twist.vx * math.cos(theta_mid) * dt
        dy = self.twist.vx * math.sin(theta_mid) * dt
        dtheta = self.twist.omega * dt

        self.pose.x += dx
        self.pose.y += dy
        self.pose.yaw += dtheta
        self.pose.normalize_yaw()

        noise_x = self.odom_noise_multiplier[0] + (random.random() - 0.5) * 0.001
        noise_y = self.odom_noise_multiplier[1] + (random.random() - 0.5) * 0.001
        noise_yaw = self.odom_noise_multiplier[2] + (random.random() - 0.5) * 0.001

        odom_theta_mid = self.odometry_pose.yaw + 0.5 * dtheta * noise_yaw
        self.odometry_pose.x += dx * noise_x * math.cos(odom_theta_mid)
        self.odometry_pose.y += dy * noise_y * math.sin(odom_theta_mid)
        self.odometry_pose.yaw += dtheta * noise_yaw
        self.odometry_pose.normalize_yaw()


class AckermannVehicle(VehicleBase):
    """
    Ackermann steering vehicle matching MVSim's VehicleAckermann & car_ackermann.
    """
    def __init__(
        self,
        name: str = "car1",
        init_pose: Optional[Pose2D] = None,
        wheelbase: float = 1.3,
        track_width: float = 0.9,
        max_steer_deg: float = 30.0,
        max_steer_rate_deg: float = 60.0,
        max_v: float = 3.0
    ):
        super().__init__(name, init_pose, mass=800.0)
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.max_steer = math.radians(max_steer_deg)
        self.max_steer_rate = math.radians(max_steer_rate_deg)
        self.max_v = max_v

        self.steer_angle: float = 0.0
        self.target_steer: float = 0.0
        self.collision_radius = math.hypot(wheelbase * 0.9, track_width * 0.6)

        self.chassis_poly = [
            (-0.4, -0.45),
            (-0.4, 0.45),
            (1.4, 0.45),
            (1.6, 0.35),
            (1.6, -0.35),
            (1.4, -0.45)
        ]

    def set_steer_and_velocity(self, target_steer_rad: float, target_v: float):
        self.target_steer = max(-self.max_steer, min(self.max_steer, target_steer_rad))
        self.target_twist.vx = max(-self.max_v, min(self.max_v, target_v))

    def step(self, dt: float):
        if dt <= 0.0:
            return

        d_steer = self.target_steer - self.steer_angle
        max_d_steer = self.max_steer_rate * dt
        if abs(d_steer) > max_d_steer:
            d_steer = math.copysign(max_d_steer, d_steer)
        self.steer_angle += d_steer

        dv = self.target_twist.vx - self.twist.vx
        max_dv = 2.0 * dt
        if abs(dv) > max_dv:
            dv = math.copysign(max_dv, dv)
        self.twist.vx += dv

        if abs(self.steer_angle) > 1e-4:
            self.twist.omega = (self.twist.vx / self.wheelbase) * math.tan(self.steer_angle)
        else:
            self.twist.omega = 0.0

        theta_mid = self.pose.yaw + 0.5 * self.twist.omega * dt
        dx = self.twist.vx * math.cos(theta_mid) * dt
        dy = self.twist.vx * math.sin(theta_mid) * dt
        dtheta = self.twist.omega * dt

        self.pose.x += dx
        self.pose.y += dy
        self.pose.yaw += dtheta
        self.pose.normalize_yaw()

        self.odometry_pose.x += dx
        self.odometry_pose.y += dy
        self.odometry_pose.yaw += dtheta
        self.odometry_pose.normalize_yaw()
