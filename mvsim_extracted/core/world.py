"""
World and physics environment simulation ported from mrpt/mvsim.
Directly corresponds to:
- modules/simulator/src/World.cpp
- modules/simulator/src/World_simul.cpp
- modules/simulator/src/Block.cpp
- modules/simulator/src/World_walls.cpp
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from .kinematics import Pose2D, VehicleBase, DifferentialVehicle, AckermannVehicle


class ObstacleWall:
    """Represents a wall segment in MVSim's world elements."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int] = (160, 160, 160)):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.color = color

    def get_segments(self) -> List[Tuple[float, float, float, float]]:
        return [(self.x1, self.y1, self.x2, self.y2)]


class NamedTarget:
    """Represents a named goal destination in the environment."""
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        description: str = "",
        dwell_time: float = 2.0
    ):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.description = description
        self.dwell_time = dwell_time
        self.reached = False
        self.reached_timestamp: Optional[float] = None

    def mark_reached(self, timestamp: float):
        self.reached = True
        self.reached_timestamp = timestamp


class ObstacleBox:
    """Represents a 2D/3D rectangular block or 3D model entity in MVSim (<block>)."""
    def __init__(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
        yaw: float = 0.0,
        color: Tuple[int, int, int] = (220, 140, 50),
        label: str = "",
        z_height: float = 2.5,
        z_elev: float = 0.0,
        model_name: str = "",
        model_path: str = ""
    ):
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = float(width)
        self.height = float(height)
        self.yaw = float(yaw)
        self.color = color
        self.label = label
        self.z_height = float(z_height)
        self.z_elev = float(z_elev)
        self.model_name = model_name
        self.model_path = model_path

    def get_corners(self) -> List[Tuple[float, float]]:
        hw = self.width / 2.0
        hh = self.height / 2.0
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)

        local_pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        world_pts = []
        for lx, ly in local_pts:
            wx = self.cx + lx * cos_y - ly * sin_y
            wy = self.cy + lx * sin_y + ly * cos_y
            world_pts.append((wx, wy))
        return world_pts

    def get_segments(self) -> List[Tuple[float, float, float, float]]:
        corners = self.get_corners()
        segments = []
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            segments.append((x1, y1, x2, y2))
        return segments


class World:
    """
    Simulation World matching MVSim's World class.
    Manages physical timestep integration, vehicles, obstacles, and collision checks.
    """
    def __init__(self, width: float = 30.0, height: float = 20.0, simul_timestep: float = 0.02):
        self.width = width
        self.height = height
        self.simul_timestep = simul_timestep
        self.simul_time = 0.0

        self.vehicles: Dict[str, VehicleBase] = {}
        self.walls: List[ObstacleWall] = []
        self.boxes: List[ObstacleBox] = []
        self.targets: List[NamedTarget] = []

        # Create outer bounding walls by default
        hw = width / 2.0
        hh = height / 2.0
        self.add_wall(-hw, -hh, hw, -hh)
        self.add_wall(hw, -hh, hw, hh)
        self.add_wall(hw, hh, -hw, hh)
        self.add_wall(-hw, hh, -hw, -hh)

    def add_vehicle(self, vehicle: VehicleBase):
        self.vehicles[vehicle.name] = vehicle

    def add_wall(self, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int] = (160, 160, 160)):
        self.walls.append(ObstacleWall(x1, y1, x2, y2, color))

    def add_box(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
        yaw: float = 0.0,
        color: Tuple[int, int, int] = (220, 140, 50),
        label: str = "",
        z_height: float = 2.5,
        z_elev: float = 0.0,
        model_name: str = "",
        model_path: str = ""
    ):
        b = ObstacleBox(cx, cy, width, height, yaw, color, label, z_height, z_elev, model_name, model_path)
        self.boxes.append(b)
        return b

    def add_target(
        self,
        name: str,
        x: float,
        y: float,
        description: str = "",
        dwell_time: float = 2.0
    ) -> NamedTarget:
        t = NamedTarget(name, x, y, description, dwell_time)
        self.targets.append(t)
        return t

    def get_all_obstacle_segments(self, exclude_vehicle: Optional[VehicleBase] = None) -> List[Tuple[float, float, float, float]]:
        """Collects all line segments from walls, boxes, and other vehicles."""
        segments: List[Tuple[float, float, float, float]] = []

        # Walls
        for w in self.walls:
            segments.extend(w.get_segments())

        # Boxes
        for b in self.boxes:
            segments.extend(b.get_segments())

        # Other vehicles (hull polygons)
        for name, veh in self.vehicles.items():
            if exclude_vehicle is not None and veh is exclude_vehicle:
                continue
            poly = veh.get_world_polygon()
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                segments.append((x1, y1, x2, y2))

        return segments

    def step(self, dt: Optional[float] = None):
        """Advances physical simulation by dt."""
        actual_dt = dt if dt is not None else self.simul_timestep

        # 1. Update vehicle dynamics
        for veh in self.vehicles.values():
            prev_x, prev_y = veh.pose.x, veh.pose.y
            veh.step(actual_dt)

            # Collision resolution against bounding walls & boxes
            hw = self.width / 2.0 - veh.collision_radius
            hh = self.height / 2.0 - veh.collision_radius
            if veh.pose.x > hw or veh.pose.x < -hw:
                veh.pose.x = max(-hw, min(hw, veh.pose.x))
                veh.twist.vx = 0.0
            if veh.pose.y > hh or veh.pose.y < -hh:
                veh.pose.y = max(-hh, min(hh, veh.pose.y))
                veh.twist.vx = 0.0

            # Simple box repulsion
            for b in self.boxes:
                dx = veh.pose.x - b.cx
                dy = veh.pose.y - b.cy
                dist = math.hypot(dx, dy)
                min_safe_dist = (min(b.width, b.height) / 2.0) + veh.collision_radius
                if dist < min_safe_dist and dist > 1e-4:
                    push = (min_safe_dist - dist)
                    veh.pose.x += (dx / dist) * push
                    veh.pose.y += (dy / dist) * push
                    veh.twist.vx *= 0.5

        self.simul_time += actual_dt

    @classmethod
    def create_preset(cls, preset_name: str = "college") -> 'World':
        """
        Creates ready-to-run worlds extracted from MVSim tutorial XML files:
        - 'college': College campus environment with departmental buildings and sequential mission targets
        - 'warehouse': modeled after demo_warehouse.world.xml with logistics stations
        - 'multi_robot': modeled after demo_2robots.world.xml
        - 'ackermann_circuit': modeled after demo_road_circuit1.world.xml
        - 'obstacle_arena': obstacle avoidance testing playground
        """
        if preset_name == "college":
            w = cls(width=34.0, height=26.0)

            # Campus Department Buildings & Architecture (with 3D heights & model properties)
            # 1. Main Administration Block (Brick red)
            w.add_box(0.0, 8.0, width=8.0, height=3.2, color=(165, 75, 75), label="ADMIN", z_height=5.5, model_name="admin_block")
            # 2. Central Library & Learning Hub (Steel blue)
            w.add_box(-9.5, 5.0, width=6.0, height=4.5, color=(70, 110, 175), label="LIBRARY", z_height=6.2, model_name="library_block")
            # 3. Computer Science & AI Tech Park (Cyan / Teal)
            w.add_box(9.5, 5.0, width=6.0, height=4.5, color=(45, 140, 145), label="CS DEPT", z_height=7.0, model_name="cs_dept_block")
            # 4. Mechanical Workshop & Robotics Fab Lab (Industrial amber)
            w.add_box(-9.5, -5.0, width=6.5, height=4.0, color=(160, 115, 50), label="WORKSHOP", z_height=5.0, model_name="workshop_block")
            # 5. Campus Cafeteria & Student Center (Warm orange)
            w.add_box(9.5, -5.0, width=6.0, height=4.0, color=(190, 105, 45), label="CAFETERIA", z_height=4.2, model_name="cafeteria_block")
            # 6. Grand Auditorium & Convention Center (Purple)
            w.add_box(0.0, -8.0, width=8.0, height=3.4, color=(125, 65, 145), label="AUDITORIUM", z_height=6.8, model_name="auditorium_block")
            # 7. Central Quad Monument & Fountain (Emerald Garden)
            w.add_box(0.0, 0.0, width=2.6, height=2.6, color=(60, 150, 100), label="MONUMENT", z_height=1.6, model_name="rock01", model_path="mvsim_model/rock01")

            # Campus 3D Model Props from mvsim_model:
            # Library: Wooden Bookshelves (wooden_bookshelf_worn)
            w.add_box(-7.0, 2.2, width=1.4, height=0.6, color=(120, 85, 45), label="SHELF", z_height=2.0, model_name="wooden_bookshelf_worn", model_path="mvsim_model/wooden_bookshelf_worn")
            w.add_box(-12.0, 2.2, width=1.4, height=0.6, color=(120, 85, 45), label="SHELF", z_height=2.0, model_name="wooden_bookshelf_worn", model_path="mvsim_model/wooden_bookshelf_worn")

            # Workshop: Tool Cart & Cargo Pallet with Crates (tool_cart, pallet, cardboard_box_01)
            w.add_box(-7.0, -2.5, width=0.9, height=0.6, color=(200, 50, 50), label="CART", z_height=1.0, model_name="tool_cart", model_path="mvsim_model/tool_cart")
            w.add_box(-12.0, -2.5, width=1.2, height=1.0, color=(170, 130, 80), label="PALLET", z_height=0.8, model_name="pallet", model_path="mvsim_model/pallet")

            # Cafeteria: Dining Table & Chairs (dining_table, outdoor_table_chair_set_01)
            w.add_box(7.0, -2.5, width=1.6, height=1.0, color=(180, 140, 90), label="TABLE", z_height=0.9, model_name="dining_table", model_path="mvsim_model/dining_table")
            w.add_box(12.0, -2.5, width=1.6, height=1.0, color=(180, 140, 90), label="PATIO", z_height=1.2, model_name="outdoor_table_chair_set_01", model_path="mvsim_model/outdoor_table_chair_set_01")

            # Auditorium: Grand Concert Piano (Piano)
            w.add_box(0.0, -5.8, width=1.8, height=1.5, color=(30, 30, 35), label="PIANO", z_height=1.1, model_name="Piano", model_path="mvsim_model/Piano")

            # Main Gate: Security Station & Radio Tower (PoliceStation, RadioTower, StopSign, ConstructionCone)
            w.add_box(3.5, -11.5, width=2.4, height=2.0, color=(140, 140, 160), label="SECURITY", z_height=3.5, model_name="PoliceStation", model_path="mvsim_model/PoliceStation")
            w.add_box(-3.5, -11.5, width=1.5, height=1.5, color=(180, 50, 50), label="TOWER", z_height=12.0, model_name="RadioTower", model_path="mvsim_model/RadioTower")
            w.add_box(1.5, -9.5, width=0.5, height=0.5, color=(220, 30, 30), label="STOP", z_height=2.2, model_name="StopSign", model_path="mvsim_model/StopSign")
            w.add_box(-1.2, -9.5, width=0.4, height=0.4, color=(240, 100, 20), label="CONE", z_height=0.7, model_name="ConstructionCone", model_path="mvsim_model/ConstructionCone")

            # Campus Walkway Boundary Walls (Courtyard fences)
            w.add_wall(-4.2, 2.5, -4.2, -2.5, color=(100, 100, 115))
            w.add_wall(4.2, 2.5, 4.2, -2.5, color=(100, 100, 115))

            # Delivery & Campus Inspection Robot: TurtleBot3 Burger from mvsim_model/turtlebot3
            campus_bot = DifferentialVehicle(
                name="campus_bot",
                init_pose=Pose2D(x=0.0, y=-11.0, yaw=math.pi / 2.0),
                wheel_separation=0.16,
                wheel_diameter=0.066,
                max_v=1.6,
                max_w=2.2
            )
            campus_bot.model_name = "turtlebot3_burger"
            campus_bot.model_path = "mvsim_model/turtlebot3"
            w.add_vehicle(campus_bot)

            # Multiple Named College Campus Targets (Sequential Tour & Delivery Mission)
            w.add_target("1. Central Library", -9.5, 1.2, "Deliver research journals & academic archives", dwell_time=2.0)
            w.add_target("2. Admin Building", 0.0, 5.0, "Dean's Office mail & registrar dispatch", dwell_time=2.0)
            w.add_target("3. Computer Science Dept", 9.5, 1.2, "AI & Robotics Lab hardware kit delivery", dwell_time=2.0)
            w.add_target("4. Campus Cafeteria", 9.5, -1.8, "Faculty meal & refreshment station", dwell_time=2.0)
            w.add_target("5. Grand Auditorium", 0.0, -5.0, "Seminar schedules & conference AV drop", dwell_time=2.0)
            w.add_target("6. Mechanical Workshop", -9.5, -1.8, "3D print prototypes & tooling transfer", dwell_time=2.0)
            w.add_target("7. Campus Main Gate", 0.0, -11.0, "Return to campus security depot & recharge", dwell_time=2.5)

            return w

        elif preset_name == "warehouse":
            w = cls(width=28.0, height=20.0)
            # Racks and aisles
            rack_y_positions = [-6.0, -2.0, 2.0, 6.0]
            rack_x_positions = [-8.0, -2.0, 4.0, 10.0]
            for ry in rack_y_positions:
                for rx in rack_x_positions:
                    w.add_box(rx, ry, width=3.2, height=1.6, color=(180, 120, 60), label="RACK")
            # Loading docks & pallets
            w.add_box(-11.0, 0.0, width=1.5, height=3.0, color=(140, 90, 40), label="DOCK")
            w.add_box(11.0, -4.0, width=1.8, height=1.8, color=(150, 100, 50), label="PALLET")
            w.add_box(11.0, 4.0, width=1.8, height=1.8, color=(150, 100, 50), label="PALLET")

            # Add primary warehouse AGV
            agv1 = DifferentialVehicle(name="agv1", init_pose=Pose2D(x=-11.0, y=-7.0, yaw=0.0))
            w.add_vehicle(agv1)

            # Targets for warehouse preset
            w.add_target("1. Loading Dock A", -11.0, -1.5, "Inbound freight intake", dwell_time=2.0)
            w.add_target("2. Aisle 2 - North Shelf", -2.0, 4.5, "High-density pallet rack", dwell_time=2.0)
            w.add_target("3. Packaging Station", 4.0, 4.5, "Automated carton sorting", dwell_time=2.0)
            w.add_target("4. Outbound Bay 4", 10.0, -2.0, "Outbound freight dock", dwell_time=2.0)
            w.add_target("5. Charging Base", -11.0, -7.0, "Inductive charging pad", dwell_time=2.5)

            return w

        elif preset_name == "multi_robot":
            w = cls(width=24.0, height=18.0)
            # Central obstacles
            w.add_box(0.0, 0.0, width=3.5, height=3.5, color=(170, 100, 70))
            w.add_box(-6.0, 4.0, width=2.0, height=2.0, color=(130, 140, 180))
            w.add_box(6.0, -4.0, width=2.0, height=2.0, color=(130, 140, 180))
            w.add_wall(-4.0, -5.0, 4.0, -5.0, color=(180, 180, 180))

            # 2 robots navigating opposite sides
            r1 = DifferentialVehicle(name="r1", init_pose=Pose2D(x=-8.0, y=-6.0, yaw=0.5))
            r2 = DifferentialVehicle(name="r2", init_pose=Pose2D(x=8.0, y=6.0, yaw=math.pi + 0.5))
            w.add_vehicle(r1)
            w.add_vehicle(r2)
            return w

        elif preset_name == "ackermann_circuit":
            w = cls(width=32.0, height=22.0)
            # Inner track barriers
            w.add_wall(-10.0, -5.0, 10.0, -5.0, color=(200, 70, 70))
            w.add_wall(10.0, -5.0, 10.0, 5.0, color=(200, 70, 70))
            w.add_wall(10.0, 5.0, -10.0, 5.0, color=(200, 70, 70))
            w.add_wall(-10.0, 5.0, -10.0, -5.0, color=(200, 70, 70))

            # Chicanes / traffic cones
            w.add_box(0.0, -8.0, width=1.5, height=1.5, color=(240, 140, 20))
            w.add_box(5.0, 8.0, width=1.5, height=1.5, color=(240, 140, 20))

            car1 = AckermannVehicle(name="car1", init_pose=Pose2D(x=-12.0, y=-8.0, yaw=0.0))
            w.add_vehicle(car1)
            return w

        else:  # default obstacle arena
            w = cls(width=20.0, height=16.0)
            w.add_box(-4.0, -2.0, 2.0, 2.0, color=(150, 120, 80))
            w.add_box(3.0, 3.0, 2.5, 1.8, color=(120, 150, 180))
            w.add_box(4.0, -3.0, 1.8, 2.5, color=(180, 120, 150))
            w.add_wall(-2.0, 4.0, 2.0, 4.0)
            r1 = DifferentialVehicle(name="r1", init_pose=Pose2D(x=0.0, y=-5.0, yaw=math.pi / 2))
            w.add_vehicle(r1)
            return w
