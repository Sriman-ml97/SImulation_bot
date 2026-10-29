"""
Main CLI entrypoint to launch MVSim simulation.
Usage:
  python run_simulation.py --preset warehouse
  python run_simulation.py --preset multi_robot --autonomous
  python run_simulation.py --world ../mvsim_repo/mvsim_tutorial/demo_warehouse.world.xml
  python run_simulation.py --headless --duration 10
"""

import argparse
import sys
import os
import time
import random
import cv2

# Ensure package directory is in Python path for direct script execution
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from .core.world import World
    from .core.xml_loader import MVSimXMLLoader
    from .visualizer import MVSimVisualizer
except (ImportError, ValueError):
    from core.world import World
    from core.xml_loader import MVSimXMLLoader
    from visualizer import MVSimVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="MVSim Extracted Python Simulator")
    parser.add_argument(
        "--preset",
        type=str,
        default="college",
        choices=["college", "warehouse", "multi_robot", "ackermann_circuit", "obstacle_arena"],
        help="Built-in preset scenario (default: college campus tour)"
    )
    parser.add_argument(
        "--world",
        type=str,
        default="",
        help="Path to an MVSim XML world file (.world.xml)"
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Start in autonomous APF obstacle avoidance mode"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window for benchmarking / testing"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Maximum run duration in seconds (0 = run until closed)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load world
    if args.world and os.path.exists(args.world):
        print(f"[MVSim] Loading XML world from: {args.world}")
        world = MVSimXMLLoader.load_from_file(args.world)
    else:
        print(f"[MVSim] Initializing preset scenario: '{args.preset}'")
        world = World.create_preset(args.preset)

    print(f"[MVSim] World initialized with {len(world.vehicles)} vehicle(s), "
          f"{len(world.boxes)} box(es), {len(world.walls)} wall(s), "
          f"{len(world.targets)} named target(s).")

    vis = MVSimVisualizer(world)
    if args.autonomous:
        vis.autonomous_mode = True

    dt = 0.033  # ~30 FPS
    start_time = time.time()
    steps = 0

    if args.headless:
        print(f"[MVSim] Running in HEADLESS mode with Autonomous SLAM & A* Navigation...")
        target_duration = args.duration if args.duration > 0 else 5.0
        while (time.time() - start_time) < target_duration:
            # Step simulation & sensors
            for name, veh in world.vehicles.items():
                scanner = vis.scanners[name]
                scanner.scan(veh, world)
                vis.imus[name].update(veh, dt)
                if vis.autonomous_mode:
                    twist = vis.navigators[name].update_and_navigate(veh, scanner, dt)
                    veh.set_target_twist(twist.vx, twist.vy, twist.omega)
                else:
                    veh.set_target_twist(0.8, 0.0, 0.2)
            world.step(dt)
            steps += 1
            time.sleep(0.005)

        elapsed = time.time() - start_time
        print(f"[MVSim] Completed {steps} headless steps in {elapsed:.2f}s ({steps/elapsed:.1f} steps/s).")
        return

    # Interactive GUI mode
    window_name = "MVSim - College Campus Simulation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            wx, wy = vis.screen_to_world(x, y)
            tgt_num = len(world.targets) + 1
            vis.add_custom_target(f"Custom Target {tgt_num}", wx, wy)
            print(f"[MVSim] Added & Set Target via Click: 'Custom Target {tgt_num}' at ({wx:.2f}, {wy:.2f})")

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n" + "=" * 65)
    print("MVSim College Campus Tour & Multi-Target Autonomous Navigation")
    print("Controls:")
    print("  [CLICK MAP] - Add / Set Custom Target anywhere on Campus")
    print("  [N]         - Manually Skip to Next College Target")
    print("  [P]         - Toggle Autonomous SLAM & A* Navigation Mode")
    print("  [G]         - Set Random Destination")
    print("  [W / S]     - Manual Throttle Forward / Reverse")
    print("  [A / D]     - Manual Steer Left / Right")
    print("  [SPACE]     - Brake")
    print("  [TAB]       - Switch Controlled Robot")
    print("  [Q / ESC]   - Quit")
    print("=" * 65 + "\n")

    keys_pressed = {}

    try:
        last_tick = time.time()
        while True:
            now = time.time()
            elapsed_frame = now - last_tick
            last_tick = now

            if args.duration > 0 and (now - start_time) >= args.duration:
                break

            # Render frame
            frame = vis.step_and_render(dt, keys_pressed)
            cv2.imshow(window_name, frame)

            # Handle key events
            key = cv2.waitKey(int(dt * 1000)) & 0xFF
            keys_pressed.clear()

            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key == ord('p') or key == ord('P'):
                vis.autonomous_mode = not vis.autonomous_mode
                mode_name = "Autonomous Campus Tour (SLAM + A*)" if vis.autonomous_mode else "Manual Teleop"
                print(f"[MVSim] Mode: {mode_name}")
            elif key == ord('n') or key == ord('N'):
                vis.skip_to_next_target()
            elif key == ord('g') or key == ord('G'):
                rx = random.uniform(-world.width * 0.35, world.width * 0.35)
                ry = random.uniform(-world.height * 0.35, world.height * 0.35)
                vis.add_custom_target(f"Random Destination", rx, ry)
            elif key == 9:  # TAB key
                veh_names = list(world.vehicles.keys())
                if veh_names:
                    idx = (veh_names.index(vis.active_vehicle_name) + 1) % len(veh_names)
                    vis.active_vehicle_name = veh_names[idx]
                    print(f"[MVSim] Switched Active Robot to: '{vis.active_vehicle_name}'")
            elif key != 255:
                keys_pressed[key] = True

    finally:
        cv2.destroyAllWindows()
        print("[MVSim] Simulation finished cleanly.")


if __name__ == "__main__":
    main()
