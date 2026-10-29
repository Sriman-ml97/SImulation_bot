/**
 * MVSim Web Simulator Engine
 * Ported and extracted from MRPT/mvsim:
 * - VehicleDynamics (VehicleDifferential.cpp, VehicleAckermann.cpp)
 * - LaserScanner.cpp (Planar 2D LiDAR raycaster)
 * - IMU.cpp (Inertial Measurement Unit)
 * - simple-obstacle-avoidance.py (Artificial Potential Field reactive navigation)
 * - World_load_xml.cpp (MVSim XML world format)
 */

// Math and Geometry Helpers
class Pose2D {
  constructor(x = 0, y = 0, yaw = 0) {
    this.x = Number(x);
    this.y = Number(y);
    this.yaw = Number(yaw);
  }

  copy() {
    return new Pose2D(this.x, this.y, this.yaw);
  }

  normalizeYaw() {
    this.yaw = (this.yaw + Math.PI) % (2 * Math.PI) - Math.PI;
  }

  transformPoint(lx, ly) {
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);
    return {
      x: this.x + lx * cosY - ly * sinY,
      y: this.y + lx * sinY + ly * cosY
    };
  }

  inverseTransformPoint(gx, gy) {
    const dx = gx - this.x;
    const dy = gy - this.y;
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);
    return {
      x: dx * cosY + dy * sinY,
      y: -dx * sinY + dy * cosY
    };
  }
}

class Twist2D {
  constructor(vx = 0, vy = 0, omega = 0) {
    this.vx = Number(vx);
    this.vy = Number(vy);
    this.omega = Number(omega);
  }
}

// Named Mission Targets
class NamedTarget {
  constructor(name, x, y, description = '', dwellTime = 2.0) {
    this.name = name;
    this.x = Number(x);
    this.y = Number(y);
    this.description = description;
    this.dwellTime = Number(dwellTime);
    this.reached = false;
    this.reachedTimestamp = null;
  }

  markReached(timestamp = performance.now() / 1000) {
    this.reached = true;
    this.reachedTimestamp = timestamp;
  }
}

// Vehicles
class VehicleBase {
  constructor(name, pose = new Pose2D(), type = 'differential') {
    this.name = name;
    this.pose = pose.copy();
    this.type = type;
    this.twist = new Twist2D();
    this.targetTwist = new Twist2D();
    this.odometryPose = pose.copy();
    this.odomDriftFactors = [1.0025, 1.0018, 1.003];
    this.collisionRadius = 0.5;
    this.trajectory = [];
  }

  setTargetTwist(vx, vy, omega) {
    this.targetTwist.vx = vx;
    this.targetTwist.vy = vy;
    this.targetTwist.omega = omega;
  }
}

class DifferentialVehicle extends VehicleBase {
  constructor(name, pose = new Pose2D()) {
    super(name, pose, 'differential');
    this.wheelSeparation = 0.52;
    this.wheelRadius = 0.10;
    this.maxV = 1.6;
    this.maxW = 2.4;
    this.accelLimit = 2.8;
    this.alphaLimit = 4.2;

    this.wheelSpeedL = 0;
    this.wheelSpeedR = 0;
    this.collisionRadius = 0.42;

    this.chassisPoly = [
      { x: -0.28, y: -0.24 },
      { x: -0.28, y: 0.24 },
      { x: 0.24, y: 0.24 },
      { x: 0.35, y: 0.12 },
      { x: 0.35, y: -0.12 },
      { x: 0.24, y: -0.24 }
    ];
  }

  step(dt) {
    if (dt <= 0) return;

    // Acceleration limits
    let dv = this.targetTwist.vx - this.twist.vx;
    const maxDv = this.accelLimit * dt;
    if (Math.abs(dv) > maxDv) dv = Math.sign(dv) * maxDv;
    this.twist.vx += dv;
    this.twist.vx = Math.max(-this.maxV, Math.min(this.maxV, this.twist.vx));

    let dw = this.targetTwist.omega - this.twist.omega;
    const maxDw = this.alphaLimit * dt;
    if (Math.abs(dw) > maxDw) dw = Math.sign(dw) * maxDw;
    this.twist.omega += dw;
    this.twist.omega = Math.max(-this.maxW, Math.min(this.maxW, this.twist.omega));

    // Wheel speeds
    this.wheelSpeedR = (this.twist.vx + (this.twist.omega * this.wheelSeparation / 2.0)) / this.wheelRadius;
    this.wheelSpeedL = (this.twist.vx - (this.twist.omega * this.wheelSeparation / 2.0)) / this.wheelRadius;

    // Runge-Kutta 2 integration
    const thetaMid = this.pose.yaw + 0.5 * this.twist.omega * dt;
    const dx = this.twist.vx * Math.cos(thetaMid) * dt;
    const dy = this.twist.vx * Math.sin(thetaMid) * dt;
    const dtheta = this.twist.omega * dt;

    this.pose.x += dx;
    this.pose.y += dy;
    this.pose.yaw += dtheta;
    this.pose.normalizeYaw();

    // Noisy odometry integration
    const noiseX = this.odomDriftFactors[0] + (Math.random() - 0.5) * 0.001;
    const noiseY = this.odomDriftFactors[1] + (Math.random() - 0.5) * 0.001;
    const noiseYaw = this.odomDriftFactors[2] + (Math.random() - 0.5) * 0.001;

    const odomThetaMid = this.odometryPose.yaw + 0.5 * dtheta * noiseYaw;
    this.odometryPose.x += dx * noiseX * Math.cos(odomThetaMid);
    this.odometryPose.y += dy * noiseY * Math.sin(odomThetaMid);
    this.odometryPose.yaw += dtheta * noiseYaw;
    this.odometryPose.normalizeYaw();
  }
}

class AckermannVehicle extends VehicleBase {
  constructor(name, pose = new Pose2D()) {
    super(name, pose, 'ackermann');
    this.wheelbase = 1.35;
    this.trackWidth = 0.85;
    this.maxSteer = Math.PI * (32 / 180);
    this.maxSteerRate = Math.PI * (70 / 180);
    this.maxV = 2.4;

    this.steerAngle = 0;
    this.targetSteer = 0;
    this.collisionRadius = 0.85;

    this.chassisPoly = [
      { x: -0.35, y: -0.4 },
      { x: -0.35, y: 0.4 },
      { x: 1.35, y: 0.4 },
      { x: 1.6, y: 0.28 },
      { x: 1.6, y: -0.28 },
      { x: 1.35, y: -0.4 }
    ];
  }

  setSteerAndVelocity(steerRad, targetV) {
    this.targetSteer = Math.max(-this.maxSteer, Math.min(this.maxSteer, steerRad));
    this.targetTwist.vx = Math.max(-this.maxV, Math.min(this.maxV, targetV));
  }

  step(dt) {
    if (dt <= 0) return;

    let dSteer = this.targetSteer - this.steerAngle;
    const maxDSteer = this.maxSteerRate * dt;
    if (Math.abs(dSteer) > maxDSteer) dSteer = Math.sign(dSteer) * maxDSteer;
    this.steerAngle += dSteer;

    let dv = this.targetTwist.vx - this.twist.vx;
    const maxDv = 2.0 * dt;
    if (Math.abs(dv) > maxDv) dv = Math.sign(dv) * maxDv;
    this.twist.vx += dv;

    if (Math.abs(this.steerAngle) > 1e-4) {
      this.twist.omega = (this.twist.vx / this.wheelbase) * Math.tan(this.steerAngle);
    } else {
      this.twist.omega = 0;
    }

    const thetaMid = this.pose.yaw + 0.5 * this.twist.omega * dt;
    const dx = this.twist.vx * Math.cos(thetaMid) * dt;
    const dy = this.twist.vx * Math.sin(thetaMid) * dt;
    const dtheta = this.twist.omega * dt;

    this.pose.x += dx;
    this.pose.y += dy;
    this.pose.yaw += dtheta;
    this.pose.normalizeYaw();

    this.odometryPose.x += dx;
    this.odometryPose.y += dy;
    this.odometryPose.yaw += dtheta;
    this.odometryPose.normalizeYaw();
  }
}

// Sensors
class LaserScanner {
  constructor(name = 'laser1', fovDeg = 270, nRays = 181, maxRange = 14) {
    this.name = name;
    this.sensorOffset = new Pose2D(0.25, 0, 0);
    this.aperture = fovDeg * (Math.PI / 180);
    this.nRays = nRays;
    this.maxRange = maxRange;
    this.rangeNoise = 0.015;
    this.enableNoise = true;

    this.recomputeRayAngles();
    this.ranges = new Array(this.nRays).fill(this.maxRange);
    this.valid = new Array(this.nRays).fill(false);
    this.hitPointsWorld = [];
  }

  recomputeRayAngles() {
    this.rayAngles = [];
    if (this.nRays > 1) {
      const dAngle = this.aperture / (this.nRays - 1);
      for (let i = 0; i < this.nRays; i++) {
        this.rayAngles.push(-0.5 * this.aperture + i * dAngle);
      }
    } else {
      this.rayAngles.push(0);
    }
  }

  scan(vehicle, world) {
    const sPos = vehicle.pose.transformPoint(this.sensorOffset.x, this.sensorOffset.y);
    const sYaw = vehicle.pose.yaw + this.sensorOffset.yaw;
    const segments = world.getAllObstacleSegments(vehicle);

    this.ranges = new Array(this.nRays).fill(this.maxRange);
    this.valid = new Array(this.nRays).fill(false);
    this.hitPointsWorld = [];

    for (let i = 0; i < this.nRays; i++) {
      const ang = sYaw + this.rayAngles[i];
      const dx = Math.cos(ang);
      const dy = Math.sin(ang);
      const ex = sPos.x + dx * this.maxRange;
      const ey = sPos.y + dy * this.maxRange;

      let minDist = this.maxRange;
      let hitX = ex;
      let hitY = ey;
      let hit = false;

      for (const seg of segments) {
        const res = lineIntersection(sPos.x, sPos.y, ex, ey, seg.x1, seg.y1, seg.x2, seg.y2);
        if (res && res.dist < minDist) {
          minDist = res.dist;
          hitX = res.x;
          hitY = res.y;
          hit = true;
        }
      }

      if (hit) {
        let dist = minDist;
        if (this.enableNoise && this.rangeNoise > 0) {
          dist += (Math.random() - 0.5) * 2 * this.rangeNoise;
          dist = Math.max(0.05, Math.min(this.maxRange, dist));
        }
        this.ranges[i] = dist;
        this.valid[i] = true;
        this.hitPointsWorld.push({ x: hitX, y: hitY });
      }
    }
    return this.ranges;
  }
}

class IMUSensor {
  constructor() {
    this.ax = 0;
    this.ay = 0;
    this.wz = 0;
    this.prevVx = 0;
    this.prevVy = 0;
    this.biasZ = 0.002;
  }

  update(vehicle, dt) {
    if (dt <= 0) return;
    const dvx = (vehicle.twist.vx - this.prevVx) / dt;
    const dvy = (vehicle.twist.vy - this.prevVy) / dt;
    this.prevVx = vehicle.twist.vx;
    this.prevVy = vehicle.twist.vy;

    this.biasZ += (Math.random() - 0.5) * 0.0005;
    this.ax = dvx + (Math.random() - 0.5) * 0.05;
    this.ay = dvy + (Math.random() - 0.5) * 0.05;
    this.wz = vehicle.twist.omega + this.biasZ + (Math.random() - 0.5) * 0.02;
  }
}

// ============================================================================
// Intelligent Learning SLAM & A* Global Path Planning Engine
// ============================================================================

class WebOccupancyGrid {
  constructor(worldWidth = 32, worldHeight = 24, resolution = 0.25) {
    this.worldWidth = worldWidth;
    this.worldHeight = worldHeight;
    this.resolution = resolution;
    this.cols = Math.ceil(worldWidth / resolution);
    this.rows = Math.ceil(worldHeight / resolution);
    this.originCol = Math.floor(this.cols / 2);
    this.originRow = Math.floor(this.rows / 2);

    // -1 = Unknown (Fog of war), 0 = Free space, 100 = Obstacle
    this.grid = new Int8Array(this.cols * this.rows).fill(-1);
    this.totalCells = this.cols * this.rows;
  }

  worldToGrid(wx, wy) {
    const col = this.originCol + Math.round(wx / this.resolution);
    const row = this.originRow - Math.round(wy / this.resolution);
    return { col, row };
  }

  gridToWorld(col, row) {
    const wx = (col - this.originCol) * this.resolution;
    const wy = (this.originRow - row) * this.resolution;
    return { x: wx, y: wy };
  }

  isInBounds(col, row) {
    return col >= 0 && col < this.cols && row >= 0 && row < this.rows;
  }

  idx(col, row) {
    return row * this.cols + col;
  }

  updateFromLidar(sensorX, sensorY, hitPoints, ranges, rayAngles, sensorYaw, maxRange) {
    const s = this.worldToGrid(sensorX, sensorY);
    if (!this.isInBounds(s.col, s.row)) return;

    // 1. Carve free space along LiDAR beams
    const step = ranges.length > 100 ? 2 : 1;
    for (let i = 0; i < ranges.length; i += step) {
      const r = Math.min(ranges[i], maxRange);
      const ang = sensorYaw + rayAngles[i];
      const ex = sensorX + r * Math.cos(ang);
      const ey = sensorY + r * Math.sin(ang);
      const e = this.worldToGrid(ex, ey);

      this.rasterizeLineFree(s.col, s.row, e.col, e.row);
    }

    // 2. Mark obstacle hit points
    for (const hp of hitPoints) {
      const g = this.worldToGrid(hp.x, hp.y);
      if (this.isInBounds(g.col, g.row)) {
        this.grid[this.idx(g.col, g.row)] = 100;
      }
    }
  }

  rasterizeLineFree(x0, y0, x1, y1) {
    let dx = Math.abs(x1 - x0);
    let dy = Math.abs(y1 - y0);
    let sx = x0 < x1 ? 1 : -1;
    let sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    let cx = x0;
    let cy = y0;

    while (true) {
      if (this.isInBounds(cx, cy)) {
        const id = this.idx(cx, cy);
        if (this.grid[id] !== 100) {
          this.grid[id] = 0; // Mark free
        }
      }
      if (cx === x1 && cy === y1) break;
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        cx += sx;
      }
      if (e2 < dx) {
        err += dx;
        cy += sy;
      }
    }
  }

  getCostmap() {
    const costmap = new Uint8Array(this.cols * this.rows);
    const inflationRadius = 2; // ~0.5m clearance buffer

    // Initialize base costs: unknown = 3, free = 1, obstacle = 255
    for (let i = 0; i < this.totalCells; i++) {
      const v = this.grid[i];
      costmap[i] = (v === -1) ? 3 : ((v === 0) ? 1 : 255);
    }

    // Inflate safety buffer around obstacles
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (this.grid[this.idx(c, r)] === 100) {
          for (let dr = -inflationRadius; dr <= inflationRadius; dr++) {
            for (let dc = -inflationRadius; dc <= inflationRadius; dc++) {
              const nc = c + dc;
              const nr = r + dr;
              if (this.isInBounds(nc, nr)) {
                const nid = this.idx(nc, nr);
                if (costmap[nid] < 200) {
                  costmap[nid] = 50; // Safety buffer
                }
              }
            }
          }
        }
      }
    }

    return costmap;
  }

  getExplorationStats() {
    let explored = 0;
    let obstacles = 0;
    let free = 0;
    for (let i = 0; i < this.totalCells; i++) {
      const v = this.grid[i];
      if (v !== -1) explored++;
      if (v === 100) obstacles++;
      if (v === 0) free++;
    }
    return {
      percentExplored: (explored / this.totalCells) * 100,
      obstaclesMapped: obstacles,
      freeMapped: free
    };
  }
}

class WebAStarPlanner {
  constructor() {
    this.neighbors = [
      { dx: 1, dy: 0, cost: 1.0 },
      { dx: -1, dy: 0, cost: 1.0 },
      { dx: 0, dy: 1, cost: 1.0 },
      { dx: 0, dy: -1, cost: 1.0 },
      { dx: 1, dy: 1, cost: 1.414 },
      { dx: -1, dy: 1, cost: 1.414 },
      { dx: 1, dy: -1, cost: 1.414 },
      { dx: -1, dy: -1, cost: 1.414 }
    ];
  }

  plan(startWorld, goalWorld, occGrid) {
    const costmap = occGrid.getCostmap();
    const start = occGrid.worldToGrid(startWorld.x, startWorld.y);
    let goal = occGrid.worldToGrid(goalWorld.x, goalWorld.y);

    if (!occGrid.isInBounds(start.col, start.row) || !occGrid.isInBounds(goal.col, goal.row)) {
      return null;
    }

    // If goal is inside obstacle or buffer, search nearest accessible cell
    if (costmap[occGrid.idx(goal.col, goal.row)] >= 200) {
      let found = false;
      for (let r = 1; r <= 6 && !found; r++) {
        for (let dy = -r; dy <= r && !found; dy++) {
          for (let dx = -r; dx <= r && !found; dx++) {
            const nc = goal.col + dx;
            const nr = goal.row + dy;
            if (occGrid.isInBounds(nc, nr) && costmap[occGrid.idx(nc, nr)] < 200) {
              goal = { col: nc, row: nr };
              found = true;
            }
          }
        }
      }
      if (!found) return null;
    }

    const openSet = [{ col: start.col, row: start.row, f: 0, g: 0 }];
    const cameFrom = new Map();
    const gScore = new Map();
    const startKey = `${start.col},${start.row}`;
    gScore.set(startKey, 0);

    const visited = new Set();
    let iterations = 0;
    const maxIter = 4000;

    while (openSet.length > 0 && iterations++ < maxIter) {
      // Find lowest f
      let bestIdx = 0;
      for (let i = 1; i < openSet.length; i++) {
        if (openSet[i].f < openSet[bestIdx].f) bestIdx = i;
      }
      const current = openSet.splice(bestIdx, 1)[0];
      const currKey = `${current.col},${current.row}`;

      if (current.col === goal.col && current.row === goal.row) {
        // Reconstruct path
        const rawPath = [{ col: current.col, row: current.row }];
        let track = currKey;
        while (cameFrom.has(track)) {
          const prev = cameFrom.get(track);
          rawPath.push(prev);
          track = `${prev.col},${prev.row}`;
        }
        rawPath.reverse();

        const worldPath = rawPath.map(p => occGrid.gridToWorld(p.col, p.row));
        return this.smoothPath(worldPath, costmap, occGrid);
      }

      visited.add(currKey);

      for (const n of this.neighbors) {
        const nc = current.col + n.dx;
        const nr = current.row + n.dy;
        if (!occGrid.isInBounds(nc, nr)) continue;

        const nKey = `${nc},${nr}`;
        if (visited.has(nKey)) continue;

        const cellCost = costmap[occGrid.idx(nc, nr)];
        if (cellCost >= 200) continue; // Impassable obstacle

        const tentativeG = current.g + n.cost * (1.0 + cellCost * 0.15);
        if (!gScore.has(nKey) || tentativeG < gScore.get(nKey)) {
          cameFrom.set(nKey, { col: current.col, row: current.row });
          gScore.set(nKey, tentativeG);
          const h = Math.hypot(goal.col - nc, goal.row - nr);
          openSet.push({ col: nc, row: nr, g: tentativeG, f: tentativeG + h });
        }
      }
    }

    return null;
  }

  smoothPath(worldPath, costmap, occGrid) {
    if (worldPath.length <= 2) return worldPath;
    const smoothed = [worldPath[0]];
    let curr = 0;

    while (curr < worldPath.length - 1) {
      let furthest = curr + 1;
      for (let test = worldPath.length - 1; test > curr + 1; test--) {
        if (this.hasLineOfSight(worldPath[curr], worldPath[test], costmap, occGrid)) {
          furthest = test;
          break;
        }
      }
      smoothed.push(worldPath[furthest]);
      curr = furthest;
    }
    return smoothed;
  }

  hasLineOfSight(p1, p2, costmap, occGrid) {
    const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const steps = Math.max(2, Math.floor(dist / (occGrid.resolution * 0.5)));

    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const wx = p1.x + t * (p2.x - p1.x);
      const wy = p1.y + t * (p2.y - p1.y);
      const g = occGrid.worldToGrid(wx, wy);
      if (!occGrid.isInBounds(g.col, g.row)) return false;
      if (costmap[occGrid.idx(g.col, g.row)] >= 50) return false;
    }
    return true;
  }
}

// Intelligent Autonomous Navigator (SLAM + A* Path Planning + Local Avoidance)
class APFObstacleAvoidance {
  constructor() {
    this.vMax = 1.35;
    this.wMax = 2.2;
    this.goalTolerance = 0.45;

    this.targets = [];
    this.currentTargetIndex = 0;
    this.activeTarget = null;
    this.completedTargetsCount = 0;
    this.targetReachedNotification = null;
    this.dwellUntil = 0;

    this.goal = { x: 0, y: 0 };
    this.occGrid = new WebOccupancyGrid(36, 28, 0.25);
    this.planner = new WebAStarPlanner();
    this.activePath = [];
    this.currentWaypointIdx = 0;
    this.state = 'NAVIGATING'; // NAVIGATING, GOAL_REACHED, RECOVERING

    this.lastReplanTime = 0;
    this.lastAttrForce = { x: 0, y: 0 };
    this.lastRepForce = { x: 0, y: 0 };
    this.lastResForce = { x: 0, y: 0 };

    this.posHistory = [];
    this.recoveryUntil = 0;
  }

  setTargets(targets) {
    this.targets = Array.isArray(targets) ? [...targets] : [];
    this.currentTargetIndex = 0;
    this.completedTargetsCount = 0;
    this.dwellUntil = 0;
    this.targetReachedNotification = null;

    if (this.targets.length > 0) {
      this.activeTarget = this.targets[0];
      this.goal = { x: this.activeTarget.x, y: this.activeTarget.y };
      this.state = 'NAVIGATING';
      this.activePath = [];
      console.log(`[CAMPUS MISSION] Initialized mission with ${this.targets.length} targets. Active Goal: '${this.activeTarget.name}'`);
    } else {
      this.activeTarget = null;
      this.goal = null;
    }
  }

  addCustomTarget(name, x, y, desc = '') {
    const t = new NamedTarget(name, x, y, desc, 2.0);
    this.targets.push(t);
    if (!this.activeTarget || this.state === 'GOAL_REACHED') {
      this.activeTarget = t;
      this.goal = { x: t.x, y: t.y };
      this.state = 'NAVIGATING';
      this.activePath = [];
    }
    console.log(`[CAMPUS MISSION] Added New Target: '${name}' at (${x.toFixed(2)}, ${y.toFixed(2)})`);
    return t;
  }

  skipToNextTarget() {
    if (!this.targets.length) return;
    this.currentTargetIndex = (this.currentTargetIndex + 1) % this.targets.length;
    this.activeTarget = this.targets[this.currentTargetIndex];
    this.dwellUntil = 0;
    this.goal = { x: this.activeTarget.x, y: this.activeTarget.y };
    this.state = 'NAVIGATING';
    this.activePath = [];
    console.log(`[CAMPUS MISSION] Switched to Destination: '${this.activeTarget.name}'`);
  }

  computeTwist(vehicle, scanner, dt = 0.016) {
    const now = performance.now() / 1000;

    // 1. Update SLAM Occupancy Grid from live LiDAR
    const sensorPose = vehicle.pose.transformPoint(scanner.sensorOffset.x, scanner.sensorOffset.y);
    this.occGrid.updateFromLidar(
      sensorPose.x,
      sensorPose.y,
      scanner.hitPointsWorld,
      scanner.ranges,
      scanner.rayAngles,
      vehicle.pose.yaw + scanner.sensorOffset.yaw,
      scanner.maxRange
    );

    // Dwell handling at reached target
    if (now < this.dwellUntil) {
      return new Twist2D(0, 0, 0);
    } else if (this.dwellUntil > 0 && now >= this.dwellUntil) {
      // Dwell finished -> Set environment as next target in sequence
      this.dwellUntil = 0;
      if (this.targets.length > 0) {
        this.currentTargetIndex = (this.currentTargetIndex + 1) % this.targets.length;
        this.activeTarget = this.targets[this.currentTargetIndex];
        this.goal = { x: this.activeTarget.x, y: this.activeTarget.y };
        this.state = 'NAVIGATING';
        this.activePath = [];
        this.lastReplanTime = 0;
        console.log(`[CAMPUS MISSION] Departing for Next Department Target: '${this.activeTarget.name}'`);
      }
    }

    if (!this.goal) return new Twist2D(0, 0, 0);

    // 2. Check if Final Goal Reached
    const distToGoal = Math.hypot(this.goal.x - vehicle.pose.x, this.goal.y - vehicle.pose.y);
    if (distToGoal <= this.goalTolerance) {
      this.state = 'GOAL_REACHED';
      this.activePath = [];

      // Trigger NOTIFICATION on target arrival
      if (this.activeTarget && !this.activeTarget.reached) {
        this.activeTarget.markReached(now);
        this.completedTargetsCount++;
        const stats = this.occGrid.getExplorationStats();

        const notif = {
          name: this.activeTarget.name,
          x: this.activeTarget.x,
          y: this.activeTarget.y,
          description: this.activeTarget.description,
          timestamp: now,
          index: this.currentTargetIndex + 1,
          total: this.targets.length,
          percentExplored: stats.percentExplored
        };
        this.targetReachedNotification = notif;

        const dwell = this.activeTarget.dwellTime || 2.0;
        this.dwellUntil = now + dwell;

        const nextIdx = (this.currentTargetIndex + 1) % this.targets.length;
        const nextName = this.targets[nextIdx] ? this.targets[nextIdx].name : 'None';

        console.log('=================================================================');
        console.log(`[CAMPUS NOTIFICATION] TARGET REACHED & NOTIFIED!`);
        console.log(`  - Destination : '${this.activeTarget.name}'`);
        console.log(`  - Coordinates : (${this.activeTarget.x.toFixed(2)}m, ${this.activeTarget.y.toFixed(2)}m)`);
        console.log(`  - Mission     : ${notif.index} of ${notif.total} college targets completed`);
        console.log(`  - Campus Map  : ${notif.percentExplored.toFixed(1)}% explored (${stats.obstaclesMapped} obstacles mapped)`);
        console.log(`  - Task Details: ${this.activeTarget.description}`);
        console.log(`  - Status      : Notified target! Proceeding to next destination: '${nextName}' in ${dwell.toFixed(1)}s`);
        console.log('=================================================================');
      }

      return new Twist2D(0, 0, 0);
    }

    // 3. Check for Stuck Condition
    this.posHistory.push({ x: vehicle.pose.x, y: vehicle.pose.y, t: now });
    this.posHistory = this.posHistory.filter(p => now - p.t <= 1.5);
    if (this.posHistory.length >= 20 && this.state === 'NAVIGATING') {
      const p0 = this.posHistory[0];
      const pN = this.posHistory[this.posHistory.length - 1];
      if (Math.hypot(pN.x - p0.x, pN.y - p0.y) < 0.08 && now > this.recoveryUntil) {
        this.state = 'RECOVERING';
        this.recoveryUntil = now + 0.8;
      }
    }

    if (this.state === 'RECOVERING') {
      if (now < this.recoveryUntil) {
        return new Twist2D(-0.3, 0, 1.6);
      } else {
        this.state = 'NAVIGATING';
        this.posHistory = [];
        this.lastReplanTime = 0;
      }
    }

    // 4. Global A* Replanning through discovered corridors
    const needsReplan = !this.activePath.length || (now - this.lastReplanTime >= 0.7);
    if (needsReplan) {
      this.lastReplanTime = now;
      const planned = this.planner.plan(
        { x: vehicle.pose.x, y: vehicle.pose.y },
        this.goal,
        this.occGrid
      );
      if (planned && planned.length >= 2) {
        this.activePath = planned;
        this.currentWaypointIdx = 1;
      } else if (!this.activePath.length) {
        this.activePath = [{ x: vehicle.pose.x, y: vehicle.pose.y }, this.goal];
        this.currentWaypointIdx = 1;
      }
    }

    // 5. Waypoint progression
    while (this.currentWaypointIdx < this.activePath.length - 1) {
      const wp = this.activePath[this.currentWaypointIdx];
      if (Math.hypot(wp.x - vehicle.pose.x, wp.y - vehicle.pose.y) < 0.7) {
        this.currentWaypointIdx++;
      } else {
        break;
      }
    }

    const targetWp = (this.currentWaypointIdx < this.activePath.length)
      ? this.activePath[this.currentWaypointIdx]
      : this.goal;

    // 6. Smooth Pure Pursuit Waypoint Guidance & Steering
    const dx = targetWp.x - vehicle.pose.x;
    const dy = targetWp.y - vehicle.pose.y;
    const targetHeading = Math.atan2(dy, dx);
    let headingErr = targetHeading - vehicle.pose.yaw;
    headingErr = (headingErr + Math.PI) % (2 * Math.PI) - Math.PI;

    const alignFactor = Math.max(0, Math.cos(headingErr));
    const speedFactor = Math.min(1.0, Math.max(0.25, distToGoal / 1.5));
    let vx = this.vMax * alignFactor * speedFactor;
    let omega = 2.4 * headingErr;

    // 7. Local LiDAR Obstacle Avoidance Buffer
    for (let i = 0; i < scanner.ranges.length; i++) {
      if (!scanner.valid[i]) continue;
      const r = scanner.ranges[i];
      if (r < 0.5) {
        const ang = scanner.rayAngles[i];
        if (Math.abs(ang) < Math.PI * (60 / 180)) {
          omega += -Math.sign(ang) * (1.6 / Math.max(0.15, r));
          vx *= 0.5;
          break;
        }
      }
    }

    omega = Math.max(-this.wMax, Math.min(this.wMax, omega));
    this.lastResForce = { x: Math.cos(vehicle.pose.yaw) * vx, y: Math.sin(vehicle.pose.yaw) * vx };

    return new Twist2D(vx, 0, omega);
  }
}

// Ray-segment intersection helper
function lineIntersection(x1, y1, x2, y2, x3, y3, x4, y4) {
  const denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
  if (Math.abs(denom) < 1e-9) return null;

  const ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
  const ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;

  if (ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1) {
    const ix = x1 + ua * (x2 - x1);
    const iy = y1 + ua * (y2 - y1);
    const dist = Math.hypot(ix - x1, iy - y1);
    return { x: ix, y: iy, dist: dist };
  }
  return null;
}

// Simulation World
class World {
  constructor(width = 28, height = 20) {
    this.width = width;
    this.height = height;
    this.simTime = 0;
    this.vehicles = new Map();
    this.boxes = [];
    this.walls = [];
    this.targets = [];
    this.initDefaultWalls();
  }

  initDefaultWalls() {
    this.walls = [];
    const hw = this.width / 2;
    const hh = this.height / 2;
    this.walls.push({ x1: -hw, y1: -hh, x2: hw, y2: -hh });
    this.walls.push({ x1: hw, y1: -hh, x2: hw, y2: hh });
    this.walls.push({ x1: hw, y1: hh, x2: -hw, y2: hh });
    this.walls.push({ x1: -hw, y1: hh, x2: -hw, y2: -hh });
  }

  addBox(cx, cy, width, height, yaw = 0, color = '#2563eb', label = '') {
    this.boxes.push({ cx, cy, width, height, yaw, color, label });
  }

  addTarget(name, x, y, description = '', dwellTime = 2.0) {
    const t = new NamedTarget(name, x, y, description, dwellTime);
    this.targets.push(t);
    return t;
  }

  getAllObstacleSegments(excludeVehicle = null) {
    const segments = [...this.walls];

    // Boxes
    for (const b of this.boxes) {
      const hw = b.width / 2;
      const hh = b.height / 2;
      const cosY = Math.cos(b.yaw);
      const sinY = Math.sin(b.yaw);
      const corners = [
        { x: b.cx + (-hw * cosY - -hh * sinY), y: b.cy + (-hw * sinY + -hh * cosY) },
        { x: b.cx + (hw * cosY - -hh * sinY), y: b.cy + (hw * sinY + -hh * cosY) },
        { x: b.cx + (hw * cosY - hh * sinY), y: b.cy + (hw * sinY + hh * cosY) },
        { x: b.cx + (-hw * cosY - hh * sinY), y: b.cy + (-hw * sinY + hh * cosY) }
      ];
      for (let i = 0; i < 4; i++) {
        segments.push({
          x1: corners[i].x,
          y1: corners[i].y,
          x2: corners[(i + 1) % 4].x,
          y2: corners[(i + 1) % 4].y
        });
      }
    }

    // Other vehicles
    for (const [name, veh] of this.vehicles) {
      if (excludeVehicle && veh === excludeVehicle) continue;
      const poly = veh.chassisPoly.map(p => veh.pose.transformPoint(p.x, p.y));
      for (let i = 0; i < poly.length; i++) {
        segments.push({
          x1: poly[i].x,
          y1: poly[i].y,
          x2: poly[(i + 1) % poly.length].x,
          y2: poly[(i + 1) % poly.length].y
        });
      }
    }

    return segments;
  }

  step(dt) {
    const hw = this.width / 2;
    const hh = this.height / 2;

    for (const veh of this.vehicles.values()) {
      veh.step(dt);

      // Boundary clamp
      const r = veh.collisionRadius;
      if (veh.pose.x > hw - r) { veh.pose.x = hw - r; veh.twist.vx = 0; }
      if (veh.pose.x < -hw + r) { veh.pose.x = -hw + r; veh.twist.vx = 0; }
      if (veh.pose.y > hh - r) { veh.pose.y = hh - r; veh.twist.vx = 0; }
      if (veh.pose.y < -hh + r) { veh.pose.y = -hh + r; veh.twist.vx = 0; }

      // Box repulsion
      for (const b of this.boxes) {
        const dx = veh.pose.x - b.cx;
        const dy = veh.pose.y - b.cy;
        const dist = Math.hypot(dx, dy);
        const minSafe = Math.min(b.width, b.height) / 2 + veh.collisionRadius;
        if (dist < minSafe && dist > 1e-4) {
          const push = minSafe - dist;
          veh.pose.x += (dx / dist) * push;
          veh.pose.y += (dy / dist) * push;
          veh.twist.vx *= 0.5;
        }
      }

      // Record trajectory
      if (!veh.trajectory.length || Math.hypot(veh.trajectory[veh.trajectory.length - 1].x - veh.pose.x, veh.trajectory[veh.trajectory.length - 1].y - veh.pose.y) > 0.15) {
        veh.trajectory.push({ x: veh.pose.x, y: veh.pose.y });
        if (veh.trajectory.length > 250) veh.trajectory.shift();
      }
    }

    this.simTime += dt;
  }

  static createPreset(name) {
    if (name === 'college') {
      const w = new World(34, 26);
      // Campus Department Buildings & Architecture (with 3D heights)
      w.addBox(0.0, 8.0, 8.0, 3.2, 0, '#a54b4b', 'ADMIN');
      w.addBox(-9.5, 5.0, 6.0, 4.5, 0, '#466eaf', 'LIBRARY');
      w.addBox(9.5, 5.0, 6.0, 4.5, 0, '#2d8c91', 'CS DEPT');
      w.addBox(-9.5, -5.0, 6.5, 4.0, 0, '#a07332', 'WORKSHOP');
      w.addBox(9.5, -5.0, 6.0, 4.0, 0, '#be692d', 'CAFETERIA');
      w.addBox(0.0, -8.0, 8.0, 3.4, 0, '#7d4191', 'AUDITORIUM');
      w.addBox(0.0, 0.0, 2.6, 2.6, 0, '#3c9664', 'MONUMENT');

      // Campus 3D Model Props from mvsim_model:
      // Library Bookshelves (wooden_bookshelf_worn)
      w.addBox(-7.0, 2.2, 1.4, 0.6, 0, '#78552d', 'SHELF');
      w.addBox(-12.0, 2.2, 1.4, 0.6, 0, '#78552d', 'SHELF');

      // Workshop Tool Cart & Pallet (tool_cart, pallet)
      w.addBox(-7.0, -2.5, 0.9, 0.6, 0, '#c83232', 'CART');
      w.addBox(-12.0, -2.5, 1.2, 1.0, 0, '#aa8250', 'PALLET');

      // Cafeteria Dining Tables (dining_table, outdoor_table_chair_set_01)
      w.addBox(7.0, -2.5, 1.6, 1.0, 0, '#b48c5a', 'TABLE');
      w.addBox(12.0, -2.5, 1.6, 1.0, 0, '#b48c5a', 'PATIO');

      // Auditorium Stage Piano (Piano)
      w.addBox(0.0, -5.8, 1.8, 1.5, 0, '#1e1e23', 'PIANO');

      // Main Gate Security Station & Radio Tower (PoliceStation, RadioTower, StopSign, ConstructionCone)
      w.addBox(3.5, -11.5, 2.4, 2.0, 0, '#8c8ca0', 'SECURITY');
      w.addBox(-3.5, -11.5, 1.5, 1.5, 0, '#b43232', 'TOWER');
      w.addBox(1.5, -9.5, 0.5, 0.5, 0, '#dc1e1e', 'STOP');
      w.addBox(-1.2, -9.5, 0.4, 0.4, 0, '#f06414', 'CONE');

      // Campus Walkway Boundary Walls (Courtyard fences)
      w.walls.push({ x1: -4.2, y1: 2.5, x2: -4.2, y2: -2.5 });
      w.walls.push({ x1: 4.2, y1: 2.5, x2: 4.2, y2: -2.5 });

      // Delivery & Campus Inspection Robot: TurtleBot3 Burger from mvsim_model/turtlebot3
      const campusBot = new DifferentialVehicle('campus_bot', new Pose2D(0.0, -11.0, Math.PI / 2.0));
      campusBot.type = 'turtlebot3';
      w.vehicles.set('campus_bot', campusBot);

      // Multiple Named College Campus Targets (Sequential Tour & Delivery Mission)
      w.addTarget('1. Central Library', -9.5, 1.2, 'Deliver research journals & academic archives', 2.0);
      w.addTarget('2. Admin Building', 0.0, 5.0, "Dean's Office mail & registrar dispatch", 2.0);
      w.addTarget('3. Computer Science Dept', 9.5, 1.2, 'AI & Robotics Lab hardware kit delivery', 2.0);
      w.addTarget('4. Campus Cafeteria', 9.5, -1.8, 'Faculty meal & refreshment station', 2.0);
      w.addTarget('5. Grand Auditorium', 0.0, -5.0, 'Seminar schedules & conference AV drop', 2.0);
      w.addTarget('6. Mechanical Workshop', -9.5, -1.8, '3D print prototypes & tooling transfer', 2.0);
      w.addTarget('7. Campus Main Gate', 0.0, -11.0, 'Return to campus security depot & recharge', 2.5);

      return w;
    } else if (name === 'warehouse') {
      const w = new World(28, 20);
      const rackY = [-6, -2, 2, 6];
      const rackX = [-8, -2, 4, 10];
      for (const ry of rackY) {
        for (const rx of rackX) {
          w.addBox(rx, ry, 3.2, 1.6, 0, '#b45309', 'RACK');
        }
      }
      w.addBox(-11, 0, 1.6, 3.2, 0, '#78350f', 'DOCK');
      w.addBox(11, -4, 1.8, 1.8, 0, '#854d0e', 'PALLET');
      w.addBox(11, 4, 1.8, 1.8, 0, '#854d0e', 'PALLET');

      const agv1 = new DifferentialVehicle('agv1', new Pose2D(-11, -7, 0));
      w.vehicles.set('agv1', agv1);

      w.addTarget('1. Loading Dock A', -11.0, -1.5, 'Inbound freight intake', 2.0);
      w.addTarget('2. North Aisle Rack', -2.0, 4.5, 'High-density pallet rack', 2.0);
      w.addTarget('3. Packaging Hub', 4.0, 4.5, 'Automated carton sorting', 2.0);
      w.addTarget('4. Outbound Bay 4', 10.0, -2.0, 'Outbound freight dock', 2.0);
      w.addTarget('5. Charging Base', -11.0, -7.0, 'Inductive charging pad', 2.5);

      return w;
    } else if (name === 'multirobot') {
      const w = new World(24, 18);
      w.addBox(0, 0, 3.5, 3.5, 0, '#3b82f6', 'CENTRAL');
      w.addBox(-6, 4, 2, 2, 0, '#6366f1', 'ZONE A');
      w.addBox(6, -4, 2, 2, 0, '#6366f1', 'ZONE B');
      w.walls.push({ x1: -5, y1: -5, x2: 5, y2: -5 });

      const r1 = new DifferentialVehicle('r1', new Pose2D(-8, -6, 0.4));
      const r2 = new DifferentialVehicle('r2', new Pose2D(8, 6, Math.PI + 0.4));
      w.vehicles.set('r1', r1);
      w.vehicles.set('r2', r2);

      w.addTarget('R1 Staging', -8.0, 4.0, 'Staging zone 1', 2.0);
      w.addTarget('R2 Staging', 8.0, -4.0, 'Staging zone 2', 2.0);

      return w;
    } else if (name === 'ackermann') {
      const w = new World(32, 22);
      w.walls.push({ x1: -10, y1: -5, x2: 10, y2: -5 });
      w.walls.push({ x1: 10, y1: -5, x2: 10, y2: 5 });
      w.walls.push({ x1: 10, y1: 5, x2: -10, y2: 5 });
      w.walls.push({ x1: -10, y1: 5, x2: -10, y2: -5 });

      w.addBox(0, -8, 1.5, 1.5, 0, '#f97316', 'PYLON');
      w.addBox(5, 8, 1.5, 1.5, 0, '#f97316', 'PYLON');

      const car1 = new AckermannVehicle('car1', new Pose2D(-12, -8, 0));
      w.vehicles.set('car1', car1);

      w.addTarget('North Checkpoint', 0.0, 8.0, 'Outer circuit timing gate', 2.0);
      w.addTarget('South Checkpoint', 0.0, -8.0, 'Pit lane entrance', 2.0);

      return w;
    } else {
      // Maze / Obstacle Arena
      const w = new World(22, 16);
      w.addBox(-4, -2, 2.5, 2.5, 0, '#10b981', 'ZONE 1');
      w.addBox(3, 3, 3.0, 2.0, 0.2, '#06b6d4', 'ZONE 2');
      w.addBox(4, -3, 2.0, 3.0, -0.3, '#ec4899', 'ZONE 3');
      w.addBox(-5, 4, 2.0, 2.0, 0, '#8b5cf6', 'ZONE 4');
      w.walls.push({ x1: -2, y1: 4, x2: 2, y2: 4 });

      const bot = new DifferentialVehicle('bot1', new Pose2D(0, -5, Math.PI / 2));
      w.vehicles.set('bot1', bot);

      w.addTarget('Maze Exit', 5.0, 5.0, 'Final beacon', 2.0);
      w.addTarget('Maze Center', 0.0, 0.0, 'Intermediate waypoint', 2.0);

      return w;
    }
  }

  toXMLString() {
    let xml = `<mvsim_world version="1.0">\n`;
    xml += `  <simul_timestep>0.02</simul_timestep>\n`;
    xml += `  <element class="horizontal_plane">\n`;
    xml += `    <x_min>${-this.width/2}</x_min><x_max>${this.width/2}</x_max>\n`;
    xml += `    <y_min>${-this.height/2}</y_min><y_max>${this.height/2}</y_max>\n`;
    xml += `  </element>\n`;

    for (let i = 0; i < this.boxes.length; i++) {
      const b = this.boxes[i];
      xml += `  <block name="box_${i+1}">\n`;
      xml += `    <init_pose>${b.cx.toFixed(2)} ${b.cy.toFixed(2)} ${(b.yaw * 180 / Math.PI).toFixed(1)}</init_pose>\n`;
      xml += `    <size_x>${b.width.toFixed(2)}</size_x><size_y>${b.height.toFixed(2)}</size_y>\n`;
      xml += `  </block>\n`;
    }

    for (const [name, veh] of this.vehicles) {
      xml += `  <vehicle name="${name}" class="${veh.type}">\n`;
      xml += `    <init_pose>${veh.pose.x.toFixed(2)} ${veh.pose.y.toFixed(2)} ${(veh.pose.yaw * 180 / Math.PI).toFixed(1)}</init_pose>\n`;
      xml += `    <include file="../definitions/lidar2d.sensor.xml" max_range="15.0" />\n`;
      xml += `  </vehicle>\n`;
    }

    xml += `</mvsim_world>`;
    return xml;
  }
}

// App Controller & UI Binding
class MVSimApp {
  constructor() {
    this.canvas = document.getElementById('sim-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.radarCanvas = document.getElementById('radar-canvas');
    this.radarCtx = this.radarCanvas.getContext('2d');

    this.world = World.createPreset('college');
    this.activeVehicleName = 'campus_bot';
    this.sensors = new Map();
    this.imus = new Map();
    this.apfs = new Map();

    this.initSensorsForWorld();

    this.isPaused = false;
    this.simSpeedMultiplier = 1.0;
    this.activeControlMode = 'apf'; // 'teleop' or 'apf'
    this.cameraFollow = true;
    this.renderLaserBeams = true;

    // Viewport camera (Pan & Zoom)
    this.ppm = 36; // pixels per meter
    this.camX = 0;
    this.camY = 0;
    this.isDraggingCam = false;
    this.dragStart = { x: 0, y: 0 };
    this.draggedBox = null;

    // Keyboard controls
    this.keysPressed = {};

    // 3D WebGL Simulation Engine (Default Mode with mvsim_model assets)
    this.is3DMode = true;
    this.engine3D = null;
    const viewportContainer = document.querySelector('.canvas-viewport');
    if (window.MVSim3DEngine && window.THREE && viewportContainer) {
      try {
        this.engine3D = new MVSim3DEngine(this, viewportContainer);
        console.log('[MVSim] 3D WebGL College Campus Engine initialized with mvsim_model assets.');
      } catch (err) {
        console.warn('[MVSim] 3D WebGL failed, falling back to 2D:', err);
        this.is3DMode = false;
      }
    }

    this.bindEvents();
    this.resizeCanvas();
    this.updateVehicleTabs();
    this.setViewMode('3d');

    this.lastTimestamp = performance.now();
    requestAnimationFrame((t) => this.loop(t));
  }

  initSensorsForWorld() {
    this.sensors.clear();
    this.imus.clear();
    this.apfs.clear();

    for (const [name, veh] of this.world.vehicles) {
      this.sensors.set(name, new LaserScanner(name + '_laser', 270, 181, 14));
      this.imus.set(name, new IMUSensor());
      const apf = new APFObstacleAvoidance();
      apf.setTargets(this.world.targets);
      this.apfs.set(name, apf);
    }

    if (!this.world.vehicles.has(this.activeVehicleName)) {
      this.activeVehicleName = this.world.vehicles.keys().next().value;
    }
  }

  bindEvents() {
    window.addEventListener('resize', () => this.resizeCanvas());

    // Keyboard
    window.addEventListener('keydown', (e) => {
      this.keysPressed[e.key.toLowerCase()] = true;
      if (e.key === 'p' || e.key === 'P') {
        this.setMode(this.activeControlMode === 'teleop' ? 'apf' : 'teleop');
      } else if (e.key === 'n' || e.key === 'N') {
        const apf = this.apfs.get(this.activeVehicleName);
        if (apf) apf.skipToNextTarget();
      } else if (e.key === ' ') {
        e.preventDefault();
        const veh = this.world.vehicles.get(this.activeVehicleName);
        if (veh) veh.setTargetTwist(0, 0, 0);
      } else if (e.key === 'Tab') {
        e.preventDefault();
        this.cycleVehicle();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keysPressed[e.key.toLowerCase()] = false;
    });

    // Canvas Pan, Zoom, and Object Drag
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      this.ppm = Math.max(15, Math.min(90, this.ppm * zoomFactor));
    });

    this.canvas.addEventListener('mousedown', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const screenX = e.clientX - rect.left;
      const screenY = e.clientY - rect.top;
      const worldPos = this.screenToWorld(screenX, screenY);

      // Check if user clicked on any obstacle box to drag it
      for (const box of this.world.boxes) {
        if (Math.abs(worldPos.x - box.cx) < box.width / 2 && Math.abs(worldPos.y - box.cy) < box.height / 2) {
          this.draggedBox = box;
          return;
        }
      }

      // If in APF mode, clicking adds / sets a new target!
      if (this.activeControlMode === 'apf') {
        const apf = this.apfs.get(this.activeVehicleName);
        if (apf) {
          const customName = `Target ${apf.targets.length + 1}`;
          apf.addCustomTarget(customName, worldPos.x, worldPos.y, 'User custom waypoint');
          return;
        }
      }

      this.isDraggingCam = true;
      this.dragStart = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mousemove', (e) => {
      if (this.draggedBox) {
        const rect = this.canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const worldPos = this.screenToWorld(screenX, screenY);
        this.draggedBox.cx = worldPos.x;
        this.draggedBox.cy = worldPos.y;
      } else if (this.isDraggingCam) {
        const dx = e.clientX - this.dragStart.x;
        const dy = e.clientY - this.dragStart.y;
        this.camX -= dx / this.ppm;
        this.camY += dy / this.ppm;
        this.dragStart = { x: e.clientX, y: e.clientY };
        this.cameraFollow = false;
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDraggingCam = false;
      this.draggedBox = null;
    });

    // Virtual Joystick on canvas
    this.initVirtualJoystick();

    // UI Buttons
    document.getElementById('btn-preset-college')?.addEventListener('click', () => this.loadPreset('college'));
    document.getElementById('btn-preset-warehouse')?.addEventListener('click', () => this.loadPreset('warehouse'));
    document.getElementById('btn-preset-multirobot')?.addEventListener('click', () => this.loadPreset('multirobot'));
    document.getElementById('btn-preset-ackermann')?.addEventListener('click', () => this.loadPreset('ackermann'));
    document.getElementById('btn-preset-maze')?.addEventListener('click', () => this.loadPreset('maze'));

    // 3D / 2D View Mode Switcher
    document.getElementById('btn-view-3d')?.addEventListener('click', () => this.setViewMode('3d'));
    document.getElementById('btn-view-2d')?.addEventListener('click', () => this.setViewMode('2d'));

    // 3D Camera Controls
    document.querySelectorAll('.cam-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const mode = e.target.dataset.cam;
        this.setCameraMode3D(mode);
      });
    });

    document.getElementById('btn-mode-teleop').addEventListener('click', () => this.setMode('teleop'));
    document.getElementById('btn-mode-apf').addEventListener('click', () => this.setMode('apf'));

    document.getElementById('btn-pause-toggle').addEventListener('click', () => this.togglePause());
    document.getElementById('btn-reset-sim').addEventListener('click', () => this.resetSimulation());
    document.getElementById('btn-center-cam').addEventListener('click', () => {
      this.cameraFollow = true;
    });

    document.getElementById('btn-spawn-box').addEventListener('click', () => {
      const veh = this.world.vehicles.get(this.activeVehicleName);
      const px = veh ? veh.pose.x + 3 : 0;
      const py = veh ? veh.pose.y : 0;
      this.world.addBox(px, py, 2.0, 2.0, 0, '#06b6d4');
    });

    document.getElementById('btn-random-goal').addEventListener('click', () => {
      const apf = this.apfs.get(this.activeVehicleName);
      if (apf) {
        const rx = (Math.random() - 0.5) * (this.world.width * 0.7);
        const ry = (Math.random() - 0.5) * (this.world.height * 0.7);
        apf.goal = { x: rx, y: ry };
      }
    });

    // Sliders
    const sliderFov = document.getElementById('slider-lidar-fov');
    sliderFov.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      document.getElementById('val-lidar-fov').textContent = val + '°';
      for (const s of this.sensors.values()) {
        s.aperture = val * (Math.PI / 180);
        s.recomputeRayAngles();
      }
    });

    const sliderRays = document.getElementById('slider-lidar-rays');
    sliderRays.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      document.getElementById('val-lidar-rays').textContent = val + ' rays';
      for (const s of this.sensors.values()) {
        s.nRays = val;
        s.recomputeRayAngles();
      }
    });

    const sliderRange = document.getElementById('slider-lidar-range');
    sliderRange.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById('val-lidar-range').textContent = val.toFixed(1) + ' m';
      for (const s of this.sensors.values()) {
        s.maxRange = val;
      }
    });

    document.getElementById('chk-lidar-noise').addEventListener('change', (e) => {
      for (const s of this.sensors.values()) s.enableNoise = e.target.checked;
    });

    document.getElementById('chk-lidar-rays').addEventListener('change', (e) => {
      this.renderLaserBeams = e.target.checked;
    });

    document.getElementById('slider-sim-speed').addEventListener('input', (e) => {
      this.simSpeedMultiplier = parseFloat(e.target.value);
      document.getElementById('val-sim-speed').textContent = this.simSpeedMultiplier.toFixed(2) + 'x';
    });

    // Modal XML Events
    const xmlModal = document.getElementById('xml-modal');
    document.getElementById('btn-xml-modal').addEventListener('click', () => {
      document.getElementById('xml-editor-textarea').value = this.world.toXMLString();
      xmlModal.classList.add('open');
    });

    document.getElementById('btn-close-modal').addEventListener('click', () => {
      xmlModal.classList.remove('open');
    });

    document.getElementById('btn-copy-xml').addEventListener('click', () => {
      const txt = document.getElementById('xml-editor-textarea').value;
      navigator.clipboard.writeText(txt).then(() => alert('MVSim XML copied to clipboard!'));
    });

    document.getElementById('btn-load-custom-xml').addEventListener('click', () => {
      const txt = document.getElementById('xml-editor-textarea').value;
      this.parseAndLoadXML(txt);
      xmlModal.classList.remove('open');
    });
  }

  initVirtualJoystick() {
    const zone = document.getElementById('joystick-zone');
    const stick = document.getElementById('joystick-stick');
    let active = false;
    let startX = 0, startY = 0;

    const handleStart = (clientX, clientY) => {
      active = true;
      const rect = zone.getBoundingClientRect();
      startX = rect.left + rect.width / 2;
      startY = rect.top + rect.height / 2;
    };

    const handleMove = (clientX, clientY) => {
      if (!active) return;
      const dx = clientX - startX;
      const dy = clientY - startY;
      const dist = Math.hypot(dx, dy);
      const maxR = 35;
      const clampedDist = Math.min(maxR, dist);
      const angle = Math.atan2(dy, dx);

      const sx = clampedDist * Math.cos(angle);
      const sy = clampedDist * Math.sin(angle);
      stick.style.transform = `translate(${sx}px, ${sy}px)`;

      // Map joystick to Twist
      const normY = -sy / maxR; // Forward/back
      const normX = -sx / maxR; // Turn left/right
      const veh = this.world.vehicles.get(this.activeVehicleName);
      if (veh) {
        if (veh instanceof AckermannVehicle) {
          veh.setSteerAndVelocity(normX * veh.maxSteer, normY * veh.maxV);
        } else {
          veh.setTargetTwist(normY * veh.maxV, 0, normX * veh.maxW);
        }
      }
    };

    const handleEnd = () => {
      if (!active) return;
      active = false;
      stick.style.transform = 'translate(0px, 0px)';
      const veh = this.world.vehicles.get(this.activeVehicleName);
      if (veh) veh.setTargetTwist(0, 0, 0);
    };

    zone.addEventListener('mousedown', (e) => handleStart(e.clientX, e.clientY));
    window.addEventListener('mousemove', (e) => handleMove(e.clientX, e.clientY));
    window.addEventListener('mouseup', handleEnd);

    zone.addEventListener('touchstart', (e) => {
      const touch = e.touches[0];
      handleStart(touch.clientX, touch.clientY);
    });
    window.addEventListener('touchmove', (e) => {
      const touch = e.touches[0];
      handleMove(touch.clientX, touch.clientY);
    });
    window.addEventListener('touchend', handleEnd);
  }

  resizeCanvas() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
  }

  worldToScreen(wx, wy) {
    const cx = this.canvas.width / (2 * window.devicePixelRatio);
    const cy = this.canvas.height / (2 * window.devicePixelRatio);
    const sx = cx + (wx - this.camX) * this.ppm;
    const sy = cy - (wy - this.camY) * this.ppm;
    return { x: sx * window.devicePixelRatio, y: sy * window.devicePixelRatio };
  }

  screenToWorld(sx, sy) {
    const cx = this.canvas.width / (2 * window.devicePixelRatio);
    const cy = this.canvas.height / (2 * window.devicePixelRatio);
    const wx = this.camX + (sx - cx) / this.ppm;
    const wy = this.camY - (sy - cy) / this.ppm;
    return { x: wx, y: wy };
  }

  setViewMode(mode) {
    this.is3DMode = (mode === '3d');
    const canvas2D = document.getElementById('sim-canvas');
    const canvas3D = document.getElementById('sim-canvas-3d');
    const camBar = document.getElementById('cam-controls-3d');
    const btn3D = document.getElementById('btn-view-3d');
    const btn2D = document.getElementById('btn-view-2d');

    if (this.is3DMode) {
      if (canvas2D) canvas2D.style.display = 'none';
      if (canvas3D) canvas3D.style.display = 'block';
      if (camBar) camBar.style.display = 'flex';
      btn3D?.classList.add('active');
      btn2D?.classList.remove('active');
      if (this.engine3D) this.engine3D.enabled = true;
    } else {
      if (canvas2D) canvas2D.style.display = 'block';
      if (canvas3D) canvas3D.style.display = 'none';
      if (camBar) camBar.style.display = 'none';
      btn2D?.classList.add('active');
      btn3D?.classList.remove('active');
      if (this.engine3D) this.engine3D.enabled = false;
    }
  }

  setCameraMode3D(mode) {
    if (this.engine3D) {
      this.engine3D.setCameraMode(mode);
      document.querySelectorAll('.cam-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.cam === mode);
      });
    }
  }

  loadPreset(name) {
    document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
    document.getElementById(`btn-preset-${name}`)?.classList.add('active');

    this.world = World.createPreset(name);
    this.activeVehicleName = this.world.vehicles.keys().next().value;
    this.initSensorsForWorld();
    this.updateVehicleTabs();
    this.cameraFollow = true;

    if (this.engine3D) {
      this.engine3D.buildTargetBeacons();
    }
  }

  setMode(mode) {
    this.activeControlMode = mode;
    document.getElementById('btn-mode-teleop').classList.toggle('active', mode === 'teleop');
    document.getElementById('btn-mode-apf').classList.toggle('active', mode === 'apf');
  }

  togglePause() {
    this.isPaused = !this.isPaused;
    document.getElementById('btn-pause-text').textContent = this.isPaused ? 'Resume' : 'Pause';
    document.getElementById('sim-status-label').textContent = this.isPaused ? 'PAUSED' : 'SIMULATING (60 FPS)';
  }

  resetSimulation() {
    for (const veh of this.world.vehicles.values()) {
      veh.pose = new Pose2D(0, 0, 0);
      veh.odometryPose = new Pose2D(0, 0, 0);
      veh.setTargetTwist(0, 0, 0);
      veh.trajectory = [];
    }
    this.cameraFollow = true;
  }

  cycleVehicle() {
    const names = Array.from(this.world.vehicles.keys());
    if (!names.length) return;
    const idx = (names.indexOf(this.activeVehicleName) + 1) % names.length;
    this.selectVehicle(names[idx]);
  }

  selectVehicle(name) {
    this.activeVehicleName = name;
    this.updateVehicleTabs();
  }

  updateVehicleTabs() {
    const container = document.getElementById('vehicle-tabs-container');
    container.innerHTML = '';
    for (const [name, veh] of this.world.vehicles) {
      const btn = document.createElement('button');
      btn.className = `veh-tab-btn ${name === this.activeVehicleName ? 'active' : ''}`;
      btn.textContent = name.toUpperCase();
      btn.addEventListener('click', () => this.selectVehicle(name));
      container.appendChild(btn);
    }
    const cur = this.world.vehicles.get(this.activeVehicleName);
    if (cur) {
      document.getElementById('badge-veh-class').textContent = cur.type.toUpperCase();
    }
  }

  parseAndLoadXML(xmlText) {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xmlText, 'text/xml');
      const root = doc.querySelector('mvsim_world');
      if (!root) throw new Error('Root <mvsim_world> tag missing');

      const newWorld = new World(30, 20);
      newWorld.vehicles.clear();

      // Parse blocks
      const blocks = doc.querySelectorAll('block');
      for (const b of blocks) {
        const poseStr = (b.querySelector('init_pose')?.textContent || '0 0 0').trim().split(/\s+/);
        const cx = parseFloat(poseStr[0]) || 0;
        const cy = parseFloat(poseStr[1]) || 0;
        const yaw = (parseFloat(poseStr[2]) || 0) * (Math.PI / 180);
        const sx = parseFloat(b.querySelector('size_x')?.textContent) || 2.0;
        const sy = parseFloat(b.querySelector('size_y')?.textContent) || 2.0;
        newWorld.addBox(cx, cy, sx, sy, yaw);
      }

      // Parse vehicles
      const vehs = doc.querySelectorAll('vehicle');
      for (const v of vehs) {
        const vName = v.getAttribute('name') || 'robot';
        const vClass = v.getAttribute('class') || 'differential';
        const poseStr = (v.querySelector('init_pose')?.textContent || '0 0 0').trim().split(/\s+/);
        const px = parseFloat(poseStr[0]) || 0;
        const py = parseFloat(poseStr[1]) || 0;
        const pyaw = (parseFloat(poseStr[2]) || 0) * (Math.PI / 180);

        if (vClass.includes('ackermann')) {
          newWorld.vehicles.set(vName, new AckermannVehicle(vName, new Pose2D(px, py, pyaw)));
        } else {
          newWorld.vehicles.set(vName, new DifferentialVehicle(vName, new Pose2D(px, py, pyaw)));
        }
      }

      this.world = newWorld;
      this.initSensorsForWorld();
      this.updateVehicleTabs();
      alert('MVSim XML world loaded successfully!');
    } catch (err) {
      alert('Failed to parse MVSim XML: ' + err.message);
    }
  }

  processControls(dt) {
    const veh = this.world.vehicles.get(this.activeVehicleName);
    if (!veh) return;

    if (this.activeControlMode === 'apf') {
      const apf = this.apfs.get(this.activeVehicleName);
      const scanner = this.sensors.get(this.activeVehicleName);
      if (apf && scanner) {
        const twist = apf.computeTwist(veh, scanner);
        if (veh instanceof AckermannVehicle) {
          const steer = Math.atan2(twist.omega * veh.wheelbase, Math.max(0.2, Math.abs(twist.vx)));
          veh.setSteerAndVelocity(steer, twist.vx);
        } else {
          veh.setTargetTwist(twist.vx, twist.vy, twist.omega);
        }
      }
    } else {
      // Manual WASD / Arrows
      let vx = 0;
      let omega = 0;
      const vStep = veh.maxV;
      const wStep = veh.maxW;

      if (this.keysPressed['w'] || this.keysPressed['arrowup']) vx += vStep;
      if (this.keysPressed['s'] || this.keysPressed['arrowdown']) vx -= vStep * 0.7;
      if (this.keysPressed['a'] || this.keysPressed['arrowleft']) omega += wStep;
      if (this.keysPressed['d'] || this.keysPressed['arrowright']) omega -= wStep;

      if (veh instanceof AckermannVehicle) {
        let steer = 0;
        if (this.keysPressed['a'] || this.keysPressed['arrowleft']) steer += veh.maxSteer;
        if (this.keysPressed['d'] || this.keysPressed['arrowright']) steer -= veh.maxSteer;
        veh.setSteerAndVelocity(steer, vx);
      } else {
        veh.setTargetTwist(vx, 0, omega);
      }
    }
  }

  loop(timestamp) {
    const elapsed = Math.min(0.1, (timestamp - this.lastTimestamp) / 1000);
    this.lastTimestamp = timestamp;

    if (!this.isPaused) {
      const dt = elapsed * this.simSpeedMultiplier;
      this.processControls(dt);
      this.world.step(dt);

      // Update sensors
      for (const [name, veh] of this.world.vehicles) {
        const s = this.sensors.get(name);
        if (s) s.scan(veh, this.world);
        const imu = this.imus.get(name);
        if (imu) imu.update(veh, dt);
      }
    }

    // Camera follow active vehicle
    if (this.cameraFollow) {
      const curVeh = this.world.vehicles.get(this.activeVehicleName);
      if (curVeh) {
        this.camX += (curVeh.pose.x - this.camX) * 0.12;
        this.camY += (curVeh.pose.y - this.camY) * 0.12;
      }
    }

    // Render 3D WebGL Scene or 2D Canvas
    if (this.is3DMode && this.engine3D) {
      this.engine3D.update(elapsed * this.simSpeedMultiplier);
    } else {
      this.render();
    }

    // Render Radar Scope & Telemetry
    this.renderRadar();
    this.updateHUD();

    requestAnimationFrame((t) => this.loop(t));
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 1. Draw Grid
    this.renderGrid(ctx);

    // 2. Draw Discovered Map from SLAM & World Obstacles
    this.renderLearnedMap(ctx, this.apfs.get(this.activeVehicleName));
    this.renderObstacles(ctx);

    // 3. Draw Vehicle Trajectories
    this.renderTrajectories(ctx);

    // 4. Draw LiDAR Beams & Hits
    const activeVeh = this.world.vehicles.get(this.activeVehicleName);
    const scanner = this.sensors.get(this.activeVehicleName);
    if (activeVeh && scanner) {
      this.renderLiDAR(ctx, activeVeh, scanner);
    }

    // 5. Draw APF Target Goal & Vectors
    if (this.activeControlMode === 'apf' && activeVeh) {
      this.renderAPFVectors(ctx, activeVeh);
    }

    // 6. Draw Vehicles
    this.renderVehicles(ctx);

    // 7. Draw Floating Target Reached Notification Toast
    this.renderNotificationToast(ctx);
  }

  renderNotificationToast(ctx) {
    const nav = this.apfs.get(this.activeVehicleName);
    if (!nav || !nav.targetReachedNotification) return;

    const notif = nav.targetReachedNotification;
    const now = performance.now() / 1000;
    const age = now - notif.timestamp;
    if (age > 4.5) return; // Notification visible for 4.5s

    const alpha = (age < 0.3) ? (age / 0.3) : ((age > 3.8) ? (4.5 - age) / 0.7 : 1.0);
    const cx = this.canvas.width / 2;
    const bannerY = 40 * window.devicePixelRatio;
    const bannerW = Math.min(this.canvas.width * 0.85, 540 * window.devicePixelRatio);
    const bannerH = 68 * window.devicePixelRatio;

    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, alpha));

    // Toast backdrop (Glassmorphism rounded card)
    ctx.fillStyle = 'rgba(10, 22, 26, 0.94)';
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.85)';
    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.shadowColor = '#06b6d4';
    ctx.shadowBlur = 16 * window.devicePixelRatio;

    const rx = cx - bannerW / 2;
    const ry = bannerY;
    const r = 10 * window.devicePixelRatio;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(rx, ry, bannerW, bannerH, r);
    } else {
      ctx.rect(rx, ry, bannerW, bannerH);
    }
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Toast Title Line
    ctx.fillStyle = '#38bdf8';
    ctx.font = `bold ${12.5 * window.devicePixelRatio}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`* [NOTIFIED TARGET] ARRIVED AT: ${notif.name.toUpperCase()} *`, cx, ry + 10 * window.devicePixelRatio);

    // Subtitle Line 1: Coordinates & Mission Progress
    ctx.fillStyle = '#f8fafc';
    ctx.font = `${10.5 * window.devicePixelRatio}px Inter, sans-serif`;
    const remainSec = Math.max(0, (nav.dwellUntil - now)).toFixed(1);
    const progressTxt = `Mission: ${notif.index}/${notif.total} Completed | Pos: (${notif.x.toFixed(1)}m, ${notif.y.toFixed(1)}m) | Explored: ${notif.percentExplored.toFixed(1)}%`;
    ctx.fillText(progressTxt, cx, ry + 28 * window.devicePixelRatio);

    // Subtitle Line 2: Task Details & Environment Sequence Status
    ctx.fillStyle = '#10b981';
    ctx.font = `italic ${10 * window.devicePixelRatio}px Inter, sans-serif`;
    const actionTxt = remainSec > 0
      ? `Task: "${notif.description}" — Setting next environment target in ${remainSec}s...`
      : `Tour continuing to next campus department...`;
    ctx.fillText(actionTxt, cx, ry + 46 * window.devicePixelRatio);

    ctx.restore();
  }

  renderGrid(ctx) {
    const hw = Math.ceil(this.world.width / 2);
    const hh = Math.ceil(this.world.height / 2);

    ctx.lineWidth = 1;
    for (let x = -hw; x <= hw; x++) {
      const p1 = this.worldToScreen(x, -hh);
      const p2 = this.worldToScreen(x, hh);
      ctx.strokeStyle = (x === 0) ? '#334155' : '#0f172a';
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    for (let y = -hh; y <= hh; y++) {
      const p1 = this.worldToScreen(-hw, y);
      const p2 = this.worldToScreen(hw, y);
      ctx.strokeStyle = (y === 0) ? '#334155' : '#0f172a';
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  renderObstacles(ctx) {
    // Walls
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 3 * window.devicePixelRatio;
    for (const w of this.world.walls) {
      const p1 = this.worldToScreen(w.x1, w.y1);
      const p2 = this.worldToScreen(w.x2, w.y2);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Boxes
    for (const b of this.world.boxes) {
      const hw = b.width / 2;
      const hh = b.height / 2;
      const cosY = Math.cos(b.yaw);
      const sinY = Math.sin(b.yaw);
      const corners = [
        this.worldToScreen(b.cx + (-hw * cosY - -hh * sinY), b.cy + (-hw * sinY + -hh * cosY)),
        this.worldToScreen(b.cx + (hw * cosY - -hh * sinY), b.cy + (hw * sinY + -hh * cosY)),
        this.worldToScreen(b.cx + (hw * cosY - hh * sinY), b.cy + (hw * sinY + hh * cosY)),
        this.worldToScreen(b.cx + (-hw * cosY - hh * sinY), b.cy + (-hw * sinY + hh * cosY))
      ];

      ctx.fillStyle = b.color || '#1e293b';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5 * window.devicePixelRatio;

      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Building / Obstacle Label
      if (b.label) {
        const centerScreen = this.worldToScreen(b.cx, b.cy);
        ctx.fillStyle = '#ffffff';
        ctx.font = `bold ${10 * window.devicePixelRatio}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
        ctx.shadowBlur = 4 * window.devicePixelRatio;
        ctx.fillText(b.label, centerScreen.x, centerScreen.y);
        ctx.shadowBlur = 0;
      }
    }
  }

  renderTrajectories(ctx) {
    for (const [name, veh] of this.world.vehicles) {
      if (veh.trajectory.length < 2) continue;
      const isCur = name === this.activeVehicleName;
      ctx.lineWidth = (isCur ? 2 : 1) * window.devicePixelRatio;
      ctx.strokeStyle = isCur ? 'rgba(6, 182, 212, 0.4)' : 'rgba(100, 116, 139, 0.25)';

      ctx.beginPath();
      const p0 = this.worldToScreen(veh.trajectory[0].x, veh.trajectory[0].y);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < veh.trajectory.length; i++) {
        const pt = this.worldToScreen(veh.trajectory[i].x, veh.trajectory[i].y);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
    }
  }

  renderLiDAR(ctx, veh, scanner) {
    const sPos = veh.pose.transformPoint(scanner.sensorOffset.x, scanner.sensorOffset.y);
    const spScreen = this.worldToScreen(sPos.x, sPos.y);

    // Laser rays
    if (this.renderLaserBeams) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.12)';
      const step = scanner.nRays > 200 ? 3 : 2;
      for (let i = 0; i < scanner.nRays; i += step) {
        if (!scanner.valid[i]) continue;
        const r = scanner.ranges[i];
        const ang = veh.pose.yaw + scanner.sensorOffset.yaw + scanner.rayAngles[i];
        const hitScreen = this.worldToScreen(sPos.x + r * Math.cos(ang), sPos.y + r * Math.sin(ang));
        ctx.beginPath();
        ctx.moveTo(spScreen.x, spScreen.y);
        ctx.lineTo(hitScreen.x, hitScreen.y);
        ctx.stroke();
      }
    }

    // Glowing Hit Points
    ctx.fillStyle = '#38bdf8';
    ctx.shadowColor = '#06b6d4';
    ctx.shadowBlur = 6 * window.devicePixelRatio;
    for (const hp of scanner.hitPointsWorld) {
      const pt = this.worldToScreen(hp.x, hp.y);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.5 * window.devicePixelRatio, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  renderLearnedMap(ctx, nav) {
    if (!nav || !nav.occGrid) return;
    const grid = nav.occGrid;
    const total = grid.totalCells;

    // 1. Draw subtle tint for discovered obstacles
    ctx.fillStyle = 'rgba(245, 158, 11, 0.45)';
    for (let r = 0; r < grid.rows; r += 1) {
      for (let c = 0; c < grid.cols; c += 1) {
        if (grid.grid[grid.idx(c, r)] === 100) {
          const w = grid.gridToWorld(c, r);
          const sp = this.worldToScreen(w.x, w.y);
          ctx.beginPath();
          ctx.arc(sp.x, sp.y, 2.5 * window.devicePixelRatio, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
  }

  renderAPFVectors(ctx, veh) {
    const nav = this.apfs.get(this.activeVehicleName);
    if (!nav) return;

    // 1. Draw A* Planned Waypoints & Route
    if (nav.activePath && nav.activePath.length >= 2) {
      const pts = nav.activePath.map(p => this.worldToScreen(p.x, p.y));

      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5 * window.devicePixelRatio;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.stroke();

      // Waypoint nodes
      for (let i = 0; i < pts.length; i++) {
        const isTarget = (i === nav.currentWaypointIdx);
        ctx.fillStyle = isTarget ? '#38bdf8' : '#059669';
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, (isTarget ? 5 : 3) * window.devicePixelRatio, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Line from vehicle to active lookahead waypoint
      if (nav.currentWaypointIdx < pts.length) {
        const vp = this.worldToScreen(veh.pose.x, veh.pose.y);
        const tp = pts[nav.currentWaypointIdx];
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
        ctx.lineWidth = 1.5 * window.devicePixelRatio;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(vp.x, vp.y);
        ctx.lineTo(tp.x, tp.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 2. Render All Mission Targets
    if (nav.targets && nav.targets.length > 0) {
      for (let i = 0; i < nav.targets.length; i++) {
        const t = nav.targets[i];
        const tp = this.worldToScreen(t.x, t.y);
        const isActive = (t === nav.activeTarget);
        const isReached = t.reached;

        ctx.lineWidth = 2 * window.devicePixelRatio;
        if (isActive) {
          // Glowing beacon with pulsating radius
          const pulse = (Math.sin(performance.now() / 200) + 1) * 0.5;
          ctx.strokeStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, (11 + pulse * 6) * window.devicePixelRatio, 0, 2 * Math.PI);
          ctx.stroke();

          ctx.fillStyle = '#0284c7';
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, 6 * window.devicePixelRatio, 0, 2 * Math.PI);
          ctx.fill();

          // Connect vehicle to active target with dashed beacon line
          const vp = this.worldToScreen(veh.pose.x, veh.pose.y);
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
          ctx.lineWidth = 1.5 * window.devicePixelRatio;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(vp.x, vp.y);
          ctx.lineTo(tp.x, tp.y);
          ctx.stroke();
          ctx.setLineDash([]);
        } else if (isReached) {
          // Reached target (green completed badge)
          ctx.strokeStyle = '#10b981';
          ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, 9 * window.devicePixelRatio, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();

          // Checkmark
          ctx.fillStyle = '#10b981';
          ctx.font = `bold ${10 * window.devicePixelRatio}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('✓', tp.x, tp.y);
        } else {
          // Pending target
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
          ctx.lineWidth = 1.5 * window.devicePixelRatio;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, 8 * window.devicePixelRatio, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Target Name Label
        ctx.fillStyle = isActive ? '#38bdf8' : (isReached ? '#a7f3d0' : '#94a3b8');
        ctx.font = `${isActive ? 'bold ' : ''}${10 * window.devicePixelRatio}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(t.name, tp.x, tp.y - (isActive ? 16 : 12) * window.devicePixelRatio);
      }
    } else if (nav.goal) {
      // Single Goal Beacon fallback
      const gp = this.worldToScreen(nav.goal.x, nav.goal.y);
      const isReached = (nav.state === 'GOAL_REACHED');
      const col = isReached ? '#38bdf8' : '#10b981';

      ctx.strokeStyle = col;
      ctx.lineWidth = 2 * window.devicePixelRatio;
      ctx.beginPath();
      ctx.arc(gp.x, gp.y, 11 * window.devicePixelRatio, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(gp.x, gp.y, 4 * window.devicePixelRatio, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  renderVehicles(ctx) {
    for (const [name, veh] of this.world.vehicles) {
      const isCur = name === this.activeVehicleName;
      const pts = veh.chassisPoly.map(p => this.worldToScreen(veh.pose.transformPoint(p.x, p.y).x, veh.pose.transformPoint(p.x, p.y).y));

      // Chassis
      ctx.fillStyle = isCur ? '#0284c7' : '#334155';
      ctx.strokeStyle = isCur ? '#38bdf8' : '#64748b';
      ctx.lineWidth = 2 * window.devicePixelRatio;

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Heading arrow
      const vp = this.worldToScreen(veh.pose.x, veh.pose.y);
      const hp = this.worldToScreen(veh.pose.x + 0.6 * Math.cos(veh.pose.yaw), veh.pose.y + 0.6 * Math.sin(veh.pose.yaw));
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2 * window.devicePixelRatio;
      ctx.beginPath();
      ctx.moveTo(vp.x, vp.y);
      ctx.lineTo(hp.x, hp.y);
      ctx.stroke();

      // Name Label
      ctx.fillStyle = '#f8fafc';
      ctx.font = `${11 * window.devicePixelRatio}px Inter`;
      ctx.fillText(name.toUpperCase(), vp.x - 12 * window.devicePixelRatio, vp.y - 14 * window.devicePixelRatio);
    }
  }

  renderRadar() {
    const rctx = this.radarCtx;
    const w = this.radarCanvas.width;
    const h = this.radarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = w / 2 - 10;

    rctx.clearRect(0, 0, w, h);

    // Range rings
    rctx.strokeStyle = 'rgba(6, 182, 212, 0.25)';
    rctx.lineWidth = 1;
    [0.33, 0.66, 1.0].forEach(ratio => {
      rctx.beginPath();
      rctx.arc(cx, cy, maxR * ratio, 0, 2 * Math.PI);
      rctx.stroke();
    });

    // Crosshairs
    rctx.beginPath();
    rctx.moveTo(cx, 10); rctx.lineTo(cx, h - 10);
    rctx.moveTo(10, cy); rctx.lineTo(w - 10, cy);
    rctx.stroke();

    // Plot scan points on radar
    const scanner = this.sensors.get(this.activeVehicleName);
    if (!scanner) return;

    rctx.fillStyle = '#06b6d4';
    let minDist = scanner.maxRange;

    for (let i = 0; i < scanner.nRays; i++) {
      if (!scanner.valid[i]) continue;
      const r = scanner.ranges[i];
      if (r < minDist) minDist = r;
      const normR = (r / scanner.maxRange) * maxR;
      const ang = scanner.rayAngles[i] - Math.PI / 2; // top is forward
      const px = cx + normR * Math.cos(ang);
      const py = cy + normR * Math.sin(ang);

      rctx.beginPath();
      rctx.arc(px, py, 1.5, 0, 2 * Math.PI);
      rctx.fill();
    }

    document.getElementById('stat-min-dist').textContent = minDist.toFixed(2) + 'm';
  }

  updateHUD() {
    const veh = this.world.vehicles.get(this.activeVehicleName);
    if (!veh) return;

    // Pose & Twist
    document.getElementById('hud-pose').textContent =
      `X: ${veh.pose.x.toFixed(2)}m | Y: ${veh.pose.y.toFixed(2)}m | θ: ${(veh.pose.yaw * 180 / Math.PI).toFixed(1)}°`;
    document.getElementById('hud-twist').textContent =
      `V: ${veh.twist.vx.toFixed(2)} m/s | W: ${veh.twist.omega.toFixed(2)} rad/s`;

    // LiDAR hits
    const scanner = this.sensors.get(this.activeVehicleName);
    if (scanner) {
      const validCount = scanner.valid.filter(v => v).length;
      const minDist = Math.min(...scanner.ranges);
      document.getElementById('hud-lidar-hits').textContent = `${validCount}/${scanner.nRays} (Min: ${minDist.toFixed(2)}m)`;
    }

    // IMU
    const imu = this.imus.get(this.activeVehicleName);
    if (imu) {
      document.getElementById('val-ax').textContent = `${imu.ax >= 0 ? '+' : ''}${imu.ax.toFixed(2)} m/s²`;
      document.getElementById('val-ay').textContent = `${imu.ay >= 0 ? '+' : ''}${imu.ay.toFixed(2)} m/s²`;
      document.getElementById('val-wz').textContent = `${imu.wz >= 0 ? '+' : ''}${imu.wz.toFixed(2)} rad/s`;

      document.getElementById('meter-ax').style.width = Math.min(100, Math.max(0, 50 + imu.ax * 15)) + '%';
      document.getElementById('meter-ay').style.width = Math.min(100, Math.max(0, 50 + imu.ay * 15)) + '%';
      document.getElementById('meter-wz').style.width = Math.min(100, Math.max(0, 50 + imu.wz * 15)) + '%';
    }

    // Odometry Drift
    document.getElementById('drift-gt').textContent = `X: ${veh.pose.x.toFixed(2)}m, Y: ${veh.pose.y.toFixed(2)}m`;
    document.getElementById('drift-odom').textContent = `X: ${veh.odometryPose.x.toFixed(2)}m, Y: ${veh.odometryPose.y.toFixed(2)}m`;
    const driftErr = Math.hypot(veh.odometryPose.x - veh.pose.x, veh.odometryPose.y - veh.pose.y);
    document.getElementById('drift-error').textContent = `Δ ${driftErr.toFixed(2)} m`;

    if (veh instanceof DifferentialVehicle) {
      const rpmL = (veh.wheelSpeedL * 60 / (2 * Math.PI)).toFixed(1);
      const rpmR = (veh.wheelSpeedR * 60 / (2 * Math.PI)).toFixed(1);
      document.getElementById('wheel-left-rpm').textContent = rpmL;
      document.getElementById('wheel-right-rpm').textContent = rpmR;
    }

    // APF Forces & Campus Target HUD
    const apf = this.apfs.get(this.activeVehicleName);
    if (apf) {
      document.getElementById('stat-f-attr').textContent = Math.hypot(apf.lastAttrForce.x, apf.lastAttrForce.y).toFixed(2) + ' N';
      document.getElementById('stat-f-rep').textContent = Math.hypot(apf.lastRepForce.x, apf.lastRepForce.y).toFixed(2) + ' N';

      const targetPill = document.getElementById('hud-target-name');
      if (targetPill) {
        if (apf.activeTarget) {
          const curIdx = apf.currentTargetIndex + 1;
          const total = apf.targets.length;
          const now = performance.now() / 1000;
          const dwellLeft = (apf.dwellUntil > now)
            ? ` [Dwelling ${(apf.dwellUntil - now).toFixed(1)}s]`
            : '';
          targetPill.textContent = `${apf.activeTarget.name} (${curIdx}/${total})${dwellLeft}`;
        } else {
          targetPill.textContent = 'None';
        }
      }
    }
  }
}

// Bootstrap Simulator on DOM Load
window.addEventListener('DOMContentLoaded', () => {
  window.mvsimApp = new MVSimApp();
});
