"""
XML world parser and exporter for mrpt/mvsim world files (.world.xml).
Directly corresponds to:
- modules/simulator/src/World_load_xml.cpp
- modules/simulator/src/xml_utils.cpp
"""

import xml.etree.ElementTree as ET
import math
import os
from typing import Optional
from .world import World, ObstacleBox, ObstacleWall
from .kinematics import Pose2D, DifferentialVehicle, AckermannVehicle
from .sensors import LaserScanner


class MVSimXMLLoader:
    """
    Parser for MVSim .world.xml files.
    """
    @staticmethod
    def load_from_file(filepath: str) -> World:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"World file not found: {filepath}")

        tree = ET.parse(filepath)
        root = tree.getroot()
        return MVSimXMLLoader.parse_xml_root(root, os.path.dirname(filepath))

    @staticmethod
    def parse_xml_root(root: ET.Element, base_dir: str = "") -> World:
        if root.tag != "mvsim_world":
            raise ValueError(f"Expected root tag <mvsim_world>, found <{root.tag}>")

        # 1. Parse timestep
        timestep_elem = root.find("simul_timestep")
        dt = 0.02
        if timestep_elem is not None and timestep_elem.text:
            try:
                parsed_dt = float(timestep_elem.text.strip())
                if parsed_dt > 0.0:
                    dt = parsed_dt
            except ValueError:
                pass

        # 2. Estimate world size from elements
        w_width = 30.0
        w_height = 20.0
        for elem in root.findall("element"):
            elem_class = elem.attrib.get("class", "")
            if elem_class == "horizontal_plane":
                xmin = float(elem.findtext("x_min", "-15"))
                xmax = float(elem.findtext("x_max", "15"))
                ymin = float(elem.findtext("y_min", "-10"))
                ymax = float(elem.findtext("y_max", "10"))
                w_width = max(w_width, abs(xmax - xmin))
                w_height = max(w_height, abs(ymax - ymin))

        world = World(width=w_width, height=w_height, simul_timestep=dt)

        # 3. Parse blocks and obstacles
        for block in root.findall("block"):
            init_pose_str = block.findtext("init_pose", "0 0 0").strip().split()
            cx = float(init_pose_str[0]) if len(init_pose_str) > 0 else 0.0
            cy = float(init_pose_str[1]) if len(init_pose_str) > 1 else 0.0
            yaw = math.radians(float(init_pose_str[2])) if len(init_pose_str) > 2 else 0.0

            # shape/size
            sx = float(block.findtext("size_x", "1.5"))
            sy = float(block.findtext("size_y", "1.5"))
            sz = float(block.findtext("size_z", "2.5"))
            b_name = block.attrib.get("name", "")
            b_class = block.attrib.get("class", "default")
            world.add_box(cx, cy, sx, sy, yaw, label=b_name, z_height=sz, model_name=b_class)

        # 4. Parse walls
        for walls_elem in root.findall("walls"):
            for seg in walls_elem.findall("wall"):
                p1 = seg.attrib.get("p1", "0,0").split(",")
                p2 = seg.attrib.get("p2", "1,1").split(",")
                world.add_wall(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))

        # 5. Parse vehicles
        for veh_elem in root.findall("vehicle"):
            v_name = veh_elem.attrib.get("name", "robot")
            v_class = veh_elem.attrib.get("class", "differential")

            pose_str = veh_elem.findtext("init_pose", "0 0 0").strip().split()
            px = float(pose_str[0]) if len(pose_str) > 0 else 0.0
            py = float(pose_str[1]) if len(pose_str) > 1 else 0.0
            pyaw = math.radians(float(pose_str[2])) if len(pose_str) > 2 else 0.0

            if "ackermann" in v_class.lower():
                veh = AckermannVehicle(name=v_name, init_pose=Pose2D(px, py, pyaw))
            else:
                veh = DifferentialVehicle(name=v_name, init_pose=Pose2D(px, py, pyaw))

            world.add_vehicle(veh)

        # 6. Parse targets
        for tgt_elem in root.findall("target"):
            t_name = tgt_elem.attrib.get("name", "Target")
            tx = float(tgt_elem.attrib.get("x", "0.0"))
            ty = float(tgt_elem.attrib.get("y", "0.0"))
            tdesc = tgt_elem.attrib.get("description", "")
            tdwell = float(tgt_elem.attrib.get("dwell_time", "2.0"))
            world.add_target(t_name, tx, ty, tdesc, tdwell)

        return world

    @staticmethod
    def export_to_xml_string(world: World) -> str:
        """Exports the World into an MVSim compatible .world.xml string."""
        lines = [
            '<mvsim_world version="1.0">',
            f'  <simul_timestep>{world.simul_timestep:.4f}</simul_timestep>',
            '  <gui>',
            '    <cam_distance>30</cam_distance>',
            '    <refresh_fps>30</refresh_fps>',
            '  </gui>',
            '  <element class="horizontal_plane">',
            f'    <x_min>{-world.width/2:.1f}</x_min>',
            f'    <x_max>{world.width/2:.1f}</x_max>',
            f'    <y_min>{-world.height/2:.1f}</y_min>',
            f'    <y_max>{world.height/2:.1f}</y_max>',
            '  </element>'
        ]

        # Boxes
        for idx, box in enumerate(world.boxes):
            lines.append(f'  <block name="box{idx+1}">')
            lines.append(f'    <init_pose>{box.cx:.2f} {box.cy:.2f} {math.degrees(box.yaw):.1f}</init_pose>')
            lines.append(f'    <size_x>{box.width:.2f}</size_x>')
            lines.append(f'    <size_y>{box.height:.2f}</size_y>')
            lines.append('  </block>')

        # Vehicles
        for name, veh in world.vehicles.items():
            v_class = "car_ackermann" if isinstance(veh, AckermannVehicle) else "differential"
            lines.append(f'  <vehicle name="{name}" class="{v_class}">')
            lines.append(f'    <init_pose>{veh.pose.x:.2f} {veh.pose.y:.2f} {math.degrees(veh.pose.yaw):.1f}</init_pose>')
            lines.append('    <include file="../definitions/lidar2d.sensor.xml" sensor_x="0.3" sensor_z="0.2" max_range="15.0" />')
            lines.append('  </vehicle>')

        lines.append('</mvsim_world>')
        return "\n".join(lines)
