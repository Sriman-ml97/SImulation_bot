# MVSim Extracted Architecture & Interactive Simulation Engine

> **Ported and extracted from [MRPT/mvsim](https://github.com/mrpt/mvsim)**  
> Lightweight, high-performance vehicle dynamics, 2D LiDAR raycasting, IMU sensor simulation, and reactive autonomous navigation (Artificial Potential Field).

---

## 1. Overview & Extraction Mapping

This package extracts the core mathematical, dynamical, and sensory components of **MRPT/mvsim** into standalone, self-contained **Python** and **Web** environments without requiring ROS, Linux, or heavy C++ compiler dependencies.

| MVSim C++/ROS Source Component | Extracted Python Implementation | Extracted Web / JS Engine | Key Responsibilities |
|---|---|---|---|
| `modules/simulator/src/VehicleDynamics/VehicleDifferential.cpp` | [`core/kinematics.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/kinematics.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`DifferentialVehicle`) | Two-wheel differential drive, wheel speeds $\omega_L, \omega_R$, acceleration limits, odometry integration. |
| `modules/simulator/src/VehicleDynamics/VehicleAckermann.cpp` | [`core/kinematics.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/kinematics.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`AckermannVehicle`) | Bicycle model, steer angle $\delta$, steer rate limits, turning radius $R = L / \tan(\delta)$. |
| `modules/simulator/src/Sensors/LaserScanner.cpp` | [`core/sensors.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/sensors.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`LaserScanner`) | 2D planar LiDAR raycasting, FOV aperture, ray count, Gaussian noise, ray-segment intersections. |
| `modules/simulator/src/Sensors/IMU.cpp` | [`core/sensors.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/sensors.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`IMUSensor`) | Linear accelerations $a_x, a_y$, gyro yaw rate $\omega_z$, white noise & bias random-walk. |
| **Occupancy Grid Mapping (SLAM)** | [`core/occupancy_grid.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/occupancy_grid.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`WebOccupancyGrid`) | Discovers and maps environment from live LiDAR, marking free corridors and obstacle borders. |
| **Global Path Planning (A*)** | [`core/path_planner.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/path_planner.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`WebAStarPlanner`) | A* search on costmap, obstacle inflation buffer, line-of-sight trajectory shortcutting. |
| **Intelligent Learning Navigator** | [`core/intelligent_navigator.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/intelligent_navigator.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`APFObstacleAvoidance`) | Coordinates SLAM learning, global A* pathfinding, dynamic replanning, unstick recovery, and goal arrival. |
| `examples_python/simple-obstacle-avoidance.py` | [`core/apf_controller.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/apf_controller.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`APFObstacleAvoidance`) | Artificial Potential Field navigation, attractive goal pull, $1/r^2$ repulsive force from LiDAR rays. |
| `modules/simulator/src/World_load_xml.cpp` | [`core/xml_loader.py`](file:///c:/Users/tech/SG-studio/mvsim_extracted/core/xml_loader.py) | [`web/simulator.js`](file:///c:/Users/tech/SG-studio/mvsim_extracted/web/simulator.js) (`MVSimApp.parseAndLoadXML`) | Parses `<mvsim_world>` XML world descriptions, vehicles, and blocks. |
| `mvsim_tutorial/*.world.xml` | `World.create_preset(...)` | `World.createPreset(...)` | Warehouse logistics AGV, Multi-robot arena, Ackermann track, Obstacle arena. |

---

## 2. Mathematical Formulations

### Differential Drive Kinematics
Given forward speed $v$ and yaw rate $\omega$, wheel radius $r$, and separation $L$:
$$\omega_R = \frac{v + \frac{\omega L}{2}}{r}, \quad \omega_L = \frac{v - \frac{\omega L}{2}}{r}$$
World state integration (Runge-Kutta 2nd order):
$$\theta_{mid} = \theta_t + \frac{1}{2} \omega \Delta t$$
$$x_{t+\Delta t} = x_t + v \cos(\theta_{mid}) \Delta t, \quad y_{t+\Delta t} = y_t + v \sin(\theta_{mid}) \Delta t, \quad \theta_{t+\Delta t} = \theta_t + \omega \Delta t$$

### Ackermann Steering Geometry
Given wheelbase $L$ and front steer angle $\delta$:
$$R = \frac{L}{\tan(\delta)}, \quad \omega = \frac{v \tan(\delta)}{L}$$

### 2D LiDAR Raycasting & Gaussian Noise
Each ray $i \in \{0, \dots, N-1\}$ has direction:
$$\alpha_i = \phi_{sensor} + \left(i - \frac{N-1}{2}\right) \frac{\Phi}{N-1}$$
Measured range with MVSim noise model:
$$r_{meas} = r_{true} + \mathcal{N}(0, \sigma^2), \quad \sigma = \text{range\_std\_noise}$$

### Artificial Potential Field (APF) Obstacle Avoidance
1. **Attractive Force toward Goal** $\vec{g}$:
   $$\vec{F}_{attr} = K_{attr} \cdot \frac{\vec{g} - \vec{p}}{\|\vec{g} - \vec{p}\|}$$
2. **Repulsive Force from LiDAR Scan Rays**:
   $$\vec{F}_{rep} = \sum_{i=1}^{N} -\frac{w_{obs}}{r_i^2} \begin{pmatrix} \cos(\alpha_i) \\ \sin(\alpha_i) \end{pmatrix} \quad (\text{for } r_i < d_{influence})$$
3. **Control Twist Command**:
   $$\vec{F}_{res} = \vec{F}_{attr} + \vec{F}_{rep}, \quad \theta_{target} = \text{atan2}(F_{res, y}, F_{res, x})$$
   $$v_x = v_{max} \max(0, \cos(\theta_{target})), \quad \omega = K_{\omega} \cdot \theta_{target}$$

---

## 3. How to Run the Simulations

### Option A: Interactive 3D Web Simulator (Browser)
The Web Simulator runs directly in any modern browser with full **3D WebGL (Three.js)** real-time rendering, multi-camera views (Follow Robot, FPV Driver Cam, Free Orbit, Top-Down), 3D point clouds, and live telemetry instruments.

1. Start the dedicated 3D server:
   ```bash
   python run_web.py
   # Or using uv:
   uv run python run_web.py
   ```
2. Open in your browser:
   [http://localhost:8085](http://localhost:8085)

**3D College Campus Features (Powered by `mvsim_model`):**
- **Robot**: **TurtleBot3 Burger** (`mvsim_model/turtlebot3`) with animated rotating wheels, caster stabilizer, spinning RPLidar A2 turret, camera housing, and forward headlights.
- **7 Campus Departments in 3D**:
  - `1. Central Library`: 3D structure with `wooden_bookshelf_worn` rows of colorful books, study desks, and reading chairs.
  - `2. Admin Building`: Red brick classical building with `OfficeDesk`, `metal_office_desk`, `OfficeChair`, `security_camera_01`, and `ManualFireAlarmButton`.
  - `3. Computer Science Dept`: High-tech glass architecture with `shelf_big_moveai` server racks and `utility_box_01` power units.
  - `4. Campus Cafeteria`: Terrace pavilion with `outdoor_table_chair_set_01` patio dining sets, `dining_table`, `dining_chair_02`, and kitchen appliances (`Refrigerator`, `vintage_microwave`).
  - `5. Grand Auditorium`: Amphitheater with black concert `Piano` on stage and theater seating rows.
  - `6. Mechanical Workshop`: Industrial facility with red `tool_cart`, `industrial_storage_cart`, cargo `pallet`, `cardboard_box_01`, and `pallet_jack`.
  - `7. Campus Main Gate`: Gatehouse (`PoliceStation`), communications `RadioTower` with blinking red beacon, `StopSign`, and `ConstructionCone` traffic markers.
- **Campus Grounds & Landscaping**: Boulevard lined with `TropicPalmTree`, manicured `shrub_04` hedges, floral `planter_box_01`, central fountain rock garden (`rock01`, `rock02`, `rock03`), and `hanging_industrial_lamp` streetlights.
- **3D Camera Modes**:
  - `Follow 3rd Person`: Smooth chase camera tracking the robot.
  - `Robot FPV`: Driver seat first-person sensor camera view.
  - `Free Orbit`: Pitch, pan, and zoom around the campus in 3D.
  - `Top-Down Drone`: Aerial tactical bird's eye view.
- **3D Real-time LiDAR & Trajectory**:
  - 3D laser fan sweeping across ground and obstacle geometry.
  - Glowing 3D point cloud particles hovering in space.
  - 3D neon emerald tube ribbon representing the smoothed A* path.
- **Dynamic 3D Drag-and-Drop**: Click and drag any 3D obstacle, crate, or cone on the campus ground to dynamically challenge the autonomous robot in real time!
- **View Switcher**: Instant toggle between `3D View` and `2D Tactical View`.

---

### Option B: Python Desktop Visualizer (OpenCV GUI)
Run the desktop visualizer with real-time rendering, keyboard teleoperation, and autonomous SLAM + A* navigation:

```bash
cd mvsim_extracted
# Run College Campus Preset with Autonomous Tour (Default)
python run_simulation.py --preset college --autonomous

# Run 3D College Campus World utilizing mvsim_model assets
python run_simulation.py --world college_campus_3d.world.xml --autonomous

# Run Warehouse Preset with Autonomous Mode
python run_simulation.py --preset warehouse --autonomous

# Run Multi-Robot Scenario
python run_simulation.py --preset multi_robot

# Run Ackermann Car Track
python run_simulation.py --preset ackermann_circuit
```

**College Campus Preset & Multi-Target Tour:**
The robot navigates through a realistic 34m x 26m campus containing 7 labeled department buildings and a sequential mission circuit:
1. `1. Central Library` - Deliver research journals & academic archives
2. `2. Admin Building` - Dean's Office mail & registrar dispatch
3. `3. Computer Science Dept` - AI & Robotics Lab hardware kit delivery
4. `4. Campus Cafeteria` - Faculty meal & refreshment station
5. `5. Grand Auditorium` - Seminar schedules & conference AV drop
6. `6. Mechanical Workshop` - 3D print prototypes & tooling transfer
7. `7. Campus Main Gate` - Return to campus security depot & recharge

**Arrival Notification & Environment Target Sequencing:**
Upon reaching each target:
- A prominent visual notification toast banner displays at the top of the viewport with mission completion count, position, and task description.
- A formatted ASCII notification banner logs to the console.
- The robot dwells at the department (e.g. 2.0s), then automatically sets the environment target to the next destination in sequence, replanning optimal paths.

**Interactive Controls in GUI:**
- `W` / `S`: Throttle Forward / Reverse
- `A` / `D`: Steer Left / Right
- `Space`: Brake
- `P`: Toggle SLAM & A* Autonomous Navigation
- `N`: Skip to Next Campus Target in Sequence
- `G`: Set New Random Target Goal
- `Left Click`: Add Custom Named Target on Canvas
- `Tab`: Switch Active Robot
- `Q` / `Esc`: Exit

---

### Option C: Python Headless / Benchmark Mode
For continuous integration, reinforcement learning pipelines, or dataset generation:

```bash
cd mvsim_extracted
python run_simulation.py --preset college --autonomous --headless --duration 5.0
```

---

### Option D: Run Unit Tests
To verify all kinematic equations, LiDAR raycasting intersections, IMU noise models, A* path planning, and target notifications:

```bash
cd mvsim_extracted
python tests/test_simulation.py
```
*All 11 unit tests pass with 0 errors.*

