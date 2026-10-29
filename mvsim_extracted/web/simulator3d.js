/**
 * MVSim 3D WebGL Simulation Engine using Three.js
 * Renders the College Campus Environment in full 3D utilizing models from mvsim_model:
 * - TurtleBot3 Burger Robot (mvsim_model/turtlebot3) with spinning wheels & LiDAR turret
 * - 7 Campus Department Buildings with architectural 3D structures & glowing signs
 * - 3D Models: Bookshelves, Tool Carts, Dining Tables, Piano, Police Station, Radio Tower,
 *              Palm Trees, Stop Signs, Traffic Cones, Crates, Pallets, Rocks
 * - 3D LiDAR point clouds, 3D laser beams, and 3D glowing A* path ribbon
 * - Multi-camera: 3rd-person Follow, 1st-person Robot FPV, Free Orbit, Top-Down Drone
 */

class MVSim3DEngine {
  constructor(app, container) {
    this.app = app;
    this.container = container;
    this.enabled = true;
    this.cameraMode = 'follow'; // 'follow', 'fpv', 'orbit', 'topdown'

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.orbitControls = null;

    // 3D Visual Objects
    this.robotMeshGroup = null;
    this.robotWheels = [];
    this.robotLidarPuck = null;
    this.robotHeadlights = [];

    this.lidarPointsMesh = null;
    this.lidarBeamsMesh = null;
    this.pathRibbonMesh = null;
    this.targetBeaconsGroup = new THREE.Group();
    this.interactiveObstaclesGroup = new THREE.Group();
    this.campusPropsGroup = new THREE.Group();
    this.buildingsGroup = new THREE.Group();

    // Drag-and-drop raycasting
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.isDragging = false;
    this.draggedMesh = null;
    this.dragOffset = new THREE.Vector3();

    this.initThree();
    this.buildCampusEnvironment();
    this.buildTargetBeacons();
    this.setupEventListeners();
  }

  initThree() {
    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0d14);
    this.scene.fog = new THREE.FogExp2(0x0a0d14, 0.012);

    // 2. Camera
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(55, aspect, 0.1, 200);
    this.camera.position.set(0, 18, -24);

    // 3. Renderer with soft shadows
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.domElement.id = 'sim-canvas-3d';
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls (for free orbit mode)
    if (window.THREE && THREE.OrbitControls) {
      this.orbitControls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.orbitControls.enableDamping = true;
      this.orbitControls.dampingFactor = 0.06;
      this.orbitControls.maxPolarAngle = Math.PI / 2 - 0.04;
      this.orbitControls.minDistance = 2.0;
      this.orbitControls.maxDistance = 65.0;
      this.orbitControls.target.set(0, 0, 0);
    }

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xdde5f4, 0.55);
    this.scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0x90b0e0, 0x253040, 0.45);
    hemiLight.position.set(0, 40, 0);
    this.scene.add(hemiLight);

    // Sun light casting shadows
    const sunLight = new THREE.DirectionalLight(0xfff6e5, 1.25);
    sunLight.position.set(20, 35, 15);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 1.0;
    sunLight.shadow.camera.far = 80;
    const d = 26;
    sunLight.shadow.camera.left = -d;
    sunLight.shadow.camera.right = d;
    sunLight.shadow.camera.top = d;
    sunLight.shadow.camera.bottom = -d;
    sunLight.shadow.bias = -0.0005;
    this.scene.add(sunLight);

    // Add groups
    this.scene.add(this.buildingsGroup);
    this.scene.add(this.campusPropsGroup);
    this.scene.add(this.interactiveObstaclesGroup);
    this.scene.add(this.targetBeaconsGroup);

    // Build robot
    this.buildTurtleBot3();
  }

  // =========================================================================
  // 3D Campus Environment Builder (Utilizing mvsim_model specifications)
  // =========================================================================
  buildCampusEnvironment() {
    // 1. Campus Ground (Asphalt & Pavement)
    const groundGeo = new THREE.PlaneGeometry(60, 50);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x1a1e27,
      roughness: 0.88,
      metalness: 0.12
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // 2. Central Courtyard Walkway (Paved academic corridor)
    const walkwayGeo = new THREE.PlaneGeometry(9.0, 30.0);
    const walkwayMat = new THREE.MeshStandardMaterial({
      color: 0x2c3342,
      roughness: 0.72,
      metalness: 0.08
    });
    const walkway = new THREE.Mesh(walkwayGeo, walkwayMat);
    walkway.rotation.x = -Math.PI / 2;
    walkway.position.set(0, 0.01, 0);
    walkway.receiveShadow = true;
    this.scene.add(walkway);

    // East-West Academic Promenade
    const promGeo = new THREE.PlaneGeometry(36.0, 7.0);
    const prom = new THREE.Mesh(promGeo, walkwayMat);
    prom.rotation.x = -Math.PI / 2;
    prom.position.set(0, 0.012, 0);
    prom.receiveShadow = true;
    this.scene.add(prom);

    // 3. Central Quad Greenery & Lawns
    const lawnGeo = new THREE.BoxGeometry(7.2, 0.06, 7.2);
    const lawnMat = new THREE.MeshStandardMaterial({
      color: 0x1d472c,
      roughness: 0.95,
      metalness: 0.05
    });
    const lawn = new THREE.Mesh(lawnGeo, lawnMat);
    lawn.position.set(0, 0.03, 0);
    lawn.receiveShadow = true;
    this.scene.add(lawn);

    // 4. Central Fountain & Rock Monument (Centerpiece)
    this.buildFountainAndRocks(0, 0);

    // 5. Build 7 Campus Department Buildings
    this.buildDepartmentBuildings();

    // 6. Build 3D Props from mvsim_model
    this.buildModelProps();
  }

  buildFountainAndRocks(x, z) {
    const fGroup = new THREE.Group();
    fGroup.position.set(x, 0, z);

    // Stone rim
    const rimGeo = new THREE.CylinderGeometry(1.6, 1.8, 0.45, 24);
    const rimMat = new THREE.MeshStandardMaterial({ color: 0x5a6375, roughness: 0.8 });
    const rim = new THREE.Mesh(rimGeo, rimMat);
    rim.position.y = 0.225;
    rim.castShadow = true;
    rim.receiveShadow = true;
    fGroup.add(rim);

    // Water pool
    const waterGeo = new THREE.CylinderGeometry(1.48, 1.48, 0.38, 24);
    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      roughness: 0.1,
      metalness: 0.8,
      transparent: true,
      opacity: 0.85
    });
    const water = new THREE.Mesh(waterGeo, waterMat);
    water.position.y = 0.22;
    fGroup.add(water);

    // Granite boulders centerpiece (mvsim_model/rock01, rock02, rock03)
    const rockMat = new THREE.MeshStandardMaterial({ color: 0x78716c, roughness: 0.9 });
    const r1 = new THREE.Mesh(new THREE.DodecahedronGeometry(0.55, 1), rockMat);
    r1.position.set(0.1, 0.55, 0.1);
    r1.scale.set(1.1, 1.4, 0.9);
    r1.castShadow = true;
    fGroup.add(r1);

    const r2 = new THREE.Mesh(new THREE.DodecahedronGeometry(0.38, 1), rockMat);
    r2.position.set(-0.35, 0.42, 0.25);
    r2.castShadow = true;
    fGroup.add(r2);

    this.scene.add(fGroup);
  }

  // =========================================================================
  // Campus Department Buildings (3D Structures with Neon Facade Signage)
  // =========================================================================
  buildDepartmentBuildings() {
    const departments = [
      {
        id: 'ADMIN',
        title: 'ADMINISTRATION BLOCK',
        x: 0.0, z: -8.0, w: 8.0, d: 3.2, h: 5.8,
        wallColor: 0x8b3a3a, trimColor: 0xe2e8f0, signColor: 0xf43f5e
      },
      {
        id: 'LIBRARY',
        title: 'CENTRAL LIBRARY',
        x: -9.5, z: -5.0, w: 6.0, d: 4.5, h: 6.5,
        wallColor: 0x2c4a75, trimColor: 0x38bdf8, signColor: 0x06b6d4
      },
      {
        id: 'CS_DEPT',
        title: 'CS & ROBOTICS LAB',
        x: 9.5, z: -5.0, w: 6.0, d: 4.5, h: 7.2,
        wallColor: 0x1e5a5c, trimColor: 0x2dd4bf, signColor: 0x14b8a6
      },
      {
        id: 'WORKSHOP',
        title: 'MECHANICAL FAB LAB',
        x: -9.5, z: 5.0, w: 6.5, d: 4.0, h: 5.2,
        wallColor: 0x784a1e, trimColor: 0xfbbf24, signColor: 0xf59e0b
      },
      {
        id: 'CAFETERIA',
        title: 'CAMPUS CAFETERIA',
        x: 9.5, z: 5.0, w: 6.0, d: 4.0, h: 4.4,
        wallColor: 0x9a4b22, trimColor: 0xfdba74, signColor: 0xf97316
      },
      {
        id: 'AUDITORIUM',
        title: 'GRAND AUDITORIUM',
        x: 0.0, z: 8.0, w: 8.0, d: 3.4, h: 7.0,
        wallColor: 0x5a2d68, trimColor: 0xe879f9, signColor: 0xa855f7
      }
    ];

    departments.forEach(dept => {
      const bGroup = new THREE.Group();
      bGroup.position.set(dept.x, 0, dept.z);

      // Main structural volume
      const bodyGeo = new THREE.BoxGeometry(dept.w, dept.h, dept.d);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: dept.wallColor,
        roughness: 0.78,
        metalness: 0.15
      });
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      body.position.y = dept.h / 2;
      body.castShadow = true;
      body.receiveShadow = true;
      bGroup.add(body);

      // Roof cornice/trim
      const roofGeo = new THREE.BoxGeometry(dept.w + 0.5, 0.45, dept.d + 0.5);
      const roofMat = new THREE.MeshStandardMaterial({ color: dept.trimColor, roughness: 0.4 });
      const roof = new THREE.Mesh(roofGeo, roofMat);
      roof.position.y = dept.h + 0.225;
      roof.castShadow = true;
      bGroup.add(roof);

      // Glass windows (Illuminated panels)
      const winGeo = new THREE.PlaneGeometry(dept.w * 0.75, 1.4);
      const winMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        emissive: 0x1e293b,
        roughness: 0.1,
        metalness: 0.9
      });
      const winFront = new THREE.Mesh(winGeo, winMat);
      winFront.position.set(0, dept.h * 0.6, dept.d / 2 + 0.02);
      bGroup.add(winFront);

      // Department Title Signboard (Glowing neon text canvas texture)
      const signSprite = this.createDepartmentSign(dept.title, dept.signColor);
      signSprite.position.set(0, dept.h + 1.2, 0);
      bGroup.add(signSprite);

      this.buildingsGroup.add(bGroup);
    });
  }

  createDepartmentSign(text, colorHex) {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');

    // Background pill
    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
    ctx.roundRect ? ctx.roundRect(10, 10, 492, 108, 20) : ctx.rect(10, 10, 492, 108);
    ctx.fill();

    // Border
    ctx.strokeStyle = `#${colorHex.toString(16).padStart(6, '0')}`;
    ctx.lineWidth = 6;
    ctx.stroke();

    // Text
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 36px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 64);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(4.2, 1.1, 1);
    return sprite;
  }

  // =========================================================================
  // 3D Model Props from mvsim_model (Placed strategically on campus)
  // =========================================================================
  buildModelProps() {
    // 1. Library: Wooden Bookshelves (mvsim_model/wooden_bookshelf_worn)
    this.createBookshelf(-7.0, -2.2, 0);
    this.createBookshelf(-12.0, -2.2, 0);

    // 2. Workshop: Tool Cart (mvsim_model/tool_cart) & Pallets (mvsim_model/pallet)
    this.createToolCart(-7.0, 2.5);
    this.createPalletWithCrates(-12.0, 2.5);

    // 3. Cafeteria: Outdoor Dining Table & Parasol (mvsim_model/dining_table, outdoor_table_chair_set_01)
    this.createDiningSet(7.0, 2.5);
    this.createDiningSet(12.0, 2.5);

    // 4. Auditorium: Grand Concert Piano (mvsim_model/Piano)
    this.createGrandPiano(0.0, 5.8);

    // 5. Main Gate: Security Station (PoliceStation) & Radio Tower (RadioTower)
    this.createSecurityGatehouse(3.5, 11.5);
    this.createRadioTower(-3.5, 11.5);
    this.createStopSign(1.5, 9.5);
    this.createConstructionCones(-1.2, 9.5);

    // 6. Boulevard Palm Trees (mvsim_model/TropicPalmTree)
    const palmLocations = [
      [-5.0, -5.5], [5.0, -5.5],
      [-5.0, 5.5], [5.0, 5.5],
      [-14.0, 0.0], [14.0, 0.0]
    ];
    palmLocations.forEach(([px, pz]) => this.createPalmTree(px, pz));

    // 7. Street Lamps (mvsim_model/hanging_industrial_lamp)
    this.createStreetLamp(-4.2, -2.5);
    this.createStreetLamp(4.2, -2.5);
    this.createStreetLamp(-4.2, 2.5);
    this.createStreetLamp(4.2, 2.5);
  }

  // Model: Wooden Bookshelf (mvsim_model/wooden_bookshelf_worn)
  createBookshelf(x, z, yaw) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);
    group.rotation.y = yaw;

    const frameGeo = new THREE.BoxGeometry(1.4, 2.0, 0.5);
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x5c3d2e, roughness: 0.85 });
    const frame = new THREE.Mesh(frameGeo, frameMat);
    frame.position.y = 1.0;
    frame.castShadow = true;
    group.add(frame);

    // Shelf dividers & colorful book spines
    const bookColors = [0xef4444, 0x3b82f6, 0x10b981, 0xf59e0b, 0x8b5cf6];
    for (let shelf = 0; shelf < 3; shelf++) {
      for (let b = 0; b < 6; b++) {
        const bg = new THREE.BoxGeometry(0.18, 0.45, 0.38);
        const bm = new THREE.MeshStandardMaterial({ color: bookColors[(shelf * 6 + b) % bookColors.length] });
        const book = new THREE.Mesh(bg, bm);
        book.position.set(-0.5 + b * 0.2, 0.45 + shelf * 0.6, 0.06);
        group.add(book);
      }
    }
    this.campusPropsGroup.add(group);
  }

  // Model: Tool Cart (mvsim_model/tool_cart)
  createToolCart(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Red mechanics metal cart
    const cartGeo = new THREE.BoxGeometry(0.85, 0.9, 0.55);
    const cartMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, metalness: 0.65, roughness: 0.35 });
    const cart = new THREE.Mesh(cartGeo, cartMat);
    cart.position.y = 0.55;
    cart.castShadow = true;
    group.add(cart);

    // Drawers & handle
    const handleGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.45);
    const handleMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.9 });
    const handle = new THREE.Mesh(handleGeo, handleMat);
    handle.rotation.z = Math.PI / 2;
    handle.position.set(0, 0.92, -0.3);
    group.add(handle);

    this.interactiveObstaclesGroup.add(group);
  }

  // Model: Pallet with Cargo Crates (mvsim_model/pallet, cardboard_box_01, wooden_crate_01)
  createPalletWithCrates(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Wooden pallet
    const palGeo = new THREE.BoxGeometry(1.2, 0.14, 1.0);
    const palMat = new THREE.MeshStandardMaterial({ color: 0xb48248, roughness: 0.9 });
    const pal = new THREE.Mesh(palGeo, palMat);
    pal.position.y = 0.07;
    pal.castShadow = true;
    group.add(pal);

    // Stacked crates
    const crateGeo = new THREE.BoxGeometry(0.5, 0.45, 0.45);
    const crateMat = new THREE.MeshStandardMaterial({ color: 0xca8a04, roughness: 0.7 });
    const c1 = new THREE.Mesh(crateGeo, crateMat);
    c1.position.set(-0.25, 0.36, -0.2);
    c1.castShadow = true;
    group.add(c1);

    const c2 = new THREE.Mesh(crateGeo, crateMat);
    c2.position.set(0.28, 0.36, 0.15);
    c2.castShadow = true;
    group.add(c2);

    const c3 = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.35, 0.4), crateMat);
    c3.position.set(0.0, 0.76, 0.0);
    c3.castShadow = true;
    group.add(c3);

    this.interactiveObstaclesGroup.add(group);
  }

  // Model: Dining Table & Chairs (mvsim_model/dining_table, outdoor_table_chair_set_01)
  createDiningSet(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Round patio table
    const tableTopGeo = new THREE.CylinderGeometry(0.75, 0.75, 0.06, 16);
    const tableMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.3 });
    const top = new THREE.Mesh(tableTopGeo, tableMat);
    top.position.y = 0.75;
    top.castShadow = true;
    group.add(top);

    const legGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.75, 8);
    const leg = new THREE.Mesh(legGeo, new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.7 }));
    leg.position.y = 0.375;
    group.add(leg);

    // Parasol Umbrella
    const umbrellaGeo = new THREE.ConeGeometry(1.25, 0.45, 16);
    const umbrellaMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.5 });
    const umbrella = new THREE.Mesh(umbrellaGeo, umbrellaMat);
    umbrella.position.y = 2.0;
    umbrella.castShadow = true;
    group.add(umbrella);

    const poleGeo = new THREE.CylinderGeometry(0.025, 0.025, 1.4, 8);
    const pole = new THREE.Mesh(poleGeo, leg.material);
    pole.position.y = 1.35;
    group.add(pole);

    this.campusPropsGroup.add(group);
  }

  // Model: Grand Concert Piano (mvsim_model/Piano)
  createGrandPiano(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Glossy black grand piano body
    const bodyGeo = new THREE.BoxGeometry(1.6, 0.55, 1.3);
    const pianoMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.15, metalness: 0.85 });
    const body = new THREE.Mesh(bodyGeo, pianoMat);
    body.position.y = 0.65;
    body.castShadow = true;
    group.add(body);

    // Piano keyboard keys
    const keysGeo = new THREE.BoxGeometry(1.2, 0.05, 0.25);
    const keysMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
    const keys = new THREE.Mesh(keysGeo, keysMat);
    keys.position.set(0, 0.7, 0.55);
    group.add(keys);

    this.campusPropsGroup.add(group);
  }

  // Model: Security Gatehouse (mvsim_model/PoliceStation)
  createSecurityGatehouse(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Gatekeeper booth
    const boothGeo = new THREE.BoxGeometry(2.4, 3.0, 2.0);
    const boothMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.6 });
    const booth = new THREE.Mesh(boothGeo, boothMat);
    booth.position.y = 1.5;
    booth.castShadow = true;
    group.add(booth);

    // Blue security beacon on roof
    const beaconGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.25, 12);
    const beaconMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, emissive: 0x3b82f6, emissiveIntensity: 1.0 });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.y = 3.15;
    group.add(beacon);

    this.campusPropsGroup.add(group);
  }

  // Model: Radio Tower (mvsim_model/RadioTower)
  createRadioTower(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Lattice antenna mast
    const mastGeo = new THREE.CylinderGeometry(0.15, 0.85, 12.0, 4);
    const mastMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, metalness: 0.8, roughness: 0.4 });
    const mast = new THREE.Mesh(mastGeo, mastMat);
    mast.position.y = 6.0;
    mast.castShadow = true;
    group.add(mast);

    // Blinking red beacon at top
    const topLight = new THREE.PointLight(0xef4444, 2.0, 15);
    topLight.position.set(0, 12.2, 0);
    group.add(topLight);

    const bMesh = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), new THREE.MeshBasicMaterial({ color: 0xff0000 }));
    bMesh.position.set(0, 12.2, 0);
    group.add(bMesh);

    this.campusPropsGroup.add(group);
  }

  // Model: Stop Sign (mvsim_model/StopSign)
  createStopSign(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Post
    const postGeo = new THREE.CylinderGeometry(0.03, 0.03, 2.2, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8 });
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.y = 1.1;
    group.add(post);

    // Octagonal red stop sign
    const signGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.02, 8);
    const signMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.4 });
    const sign = new THREE.Mesh(signGeo, signMat);
    sign.rotation.x = Math.PI / 2;
    sign.position.y = 1.95;
    group.add(sign);

    this.interactiveObstaclesGroup.add(group);
  }

  // Model: Construction Cones (mvsim_model/ConstructionCone)
  createConstructionCones(x, z) {
    for (let i = 0; i < 2; i++) {
      const coneGeo = new THREE.ConeGeometry(0.18, 0.65, 16);
      const coneMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.5 });
      const cone = new THREE.Mesh(coneGeo, coneMat);
      cone.position.set(x, 0.325, z - i * 0.9);
      cone.castShadow = true;
      this.interactiveObstaclesGroup.add(cone);
    }
  }

  // Model: Palm Tree (mvsim_model/TropicPalmTree)
  createPalmTree(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Curved brown trunk
    const trunkGeo = new THREE.CylinderGeometry(0.18, 0.28, 4.5, 8);
    const trunkMat = new THREE.MeshStandardMaterial({ color: 0x6b4f35, roughness: 0.9 });
    const trunk = new THREE.Mesh(trunkGeo, trunkMat);
    trunk.position.y = 2.25;
    trunk.castShadow = true;
    group.add(trunk);

    // Palm leaf canopy
    const leafMat = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.7, side: THREE.DoubleSide });
    for (let a = 0; a < 8; a++) {
      const leafGeo = new THREE.PlaneGeometry(1.6, 0.55);
      const leaf = new THREE.Mesh(leafGeo, leafMat);
      leaf.position.set(0, 4.4, 0);
      leaf.rotation.y = a * (Math.PI / 4);
      leaf.rotation.x = -Math.PI / 5;
      leaf.castShadow = true;
      group.add(leaf);
    }
    this.campusPropsGroup.add(group);
  }

  // Model: Street Lamp (mvsim_model/hanging_industrial_lamp)
  createStreetLamp(x, z) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    const poleGeo = new THREE.CylinderGeometry(0.06, 0.08, 4.2, 8);
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8 });
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.y = 2.1;
    group.add(pole);

    const lampGeo = new THREE.SphereGeometry(0.2, 12, 12);
    const lampMat = new THREE.MeshStandardMaterial({ color: 0xfef08a, emissive: 0xfef08a, emissiveIntensity: 0.8 });
    const lamp = new THREE.Mesh(lampGeo, lampMat);
    lamp.position.y = 4.15;
    group.add(lamp);

    const light = new THREE.PointLight(0xfffbeb, 1.2, 12);
    light.position.y = 4.1;
    group.add(light);

    this.campusPropsGroup.add(group);
  }

  // =========================================================================
  // 3D Robot Builder: TurtleBot3 Burger (from mvsim_model/turtlebot3)
  // =========================================================================
  buildTurtleBot3() {
    this.robotMeshGroup = new THREE.Group();

    // 1. Bottom Chassis Plate (Dark industrial graphite)
    const basePlateGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.02, 24);
    const plateMat = new THREE.MeshStandardMaterial({ color: 0x22262e, roughness: 0.6, metalness: 0.4 });
    const basePlate = new THREE.Mesh(basePlateGeo, plateMat);
    basePlate.position.y = 0.05;
    basePlate.castShadow = true;
    this.robotMeshGroup.add(basePlate);

    // 2. Middle Electronics Deck
    const midPlate = new THREE.Mesh(basePlateGeo, plateMat);
    midPlate.position.y = 0.12;
    midPlate.castShadow = true;
    this.robotMeshGroup.add(midPlate);

    // 3. Top Sensor Deck
    const topPlate = new THREE.Mesh(basePlateGeo, plateMat);
    topPlate.position.y = 0.18;
    topPlate.castShadow = true;
    this.robotMeshGroup.add(topPlate);

    // Standoff pillars between decks (brass spacers)
    const spacerMat = new THREE.MeshStandardMaterial({ color: 0xd97706, metalness: 0.9, roughness: 0.2 });
    for (let i = 0; i < 4; i++) {
      const ang = i * (Math.PI / 2);
      const s1 = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.07, 8), spacerMat);
      s1.position.set(0.12 * Math.cos(ang), 0.085, 0.12 * Math.sin(ang));
      this.robotMeshGroup.add(s1);

      const s2 = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.06, 8), spacerMat);
      s2.position.set(0.12 * Math.cos(ang), 0.15, 0.12 * Math.sin(ang));
      this.robotMeshGroup.add(s2);
    }

    // 4. Dual Drive Wheels (mvsim_model/turtlebot3/tire.dae geometry)
    this.robotWheels = [];
    const wheelGeo = new THREE.CylinderGeometry(0.033, 0.033, 0.022, 16);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const rimMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.8 });

    // Left Wheel
    const leftWheelGroup = new THREE.Group();
    const lTire = new THREE.Mesh(wheelGeo, wheelMat);
    lTire.rotation.z = Math.PI / 2;
    leftWheelGroup.add(lTire);
    const lRim = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.024, 12), rimMat);
    lRim.rotation.z = Math.PI / 2;
    leftWheelGroup.add(lRim);
    leftWheelGroup.position.set(0, 0.033, 0.08); // wheel_separation = 0.16m
    leftWheelGroup.castShadow = true;
    this.robotMeshGroup.add(leftWheelGroup);
    this.robotWheels.push(leftWheelGroup);

    // Right Wheel
    const rightWheelGroup = new THREE.Group();
    const rTire = new THREE.Mesh(wheelGeo, wheelMat);
    rTire.rotation.z = Math.PI / 2;
    rightWheelGroup.add(rTire);
    const rRim = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.024, 12), rimMat);
    rRim.rotation.z = Math.PI / 2;
    rightWheelGroup.add(rRim);
    rightWheelGroup.position.set(0, 0.033, -0.08);
    rightWheelGroup.castShadow = true;
    this.robotMeshGroup.add(rightWheelGroup);
    this.robotWheels.push(rightWheelGroup);

    // Caster Ball (Stabilizer)
    const casterGeo = new THREE.SphereGeometry(0.015, 12, 12);
    const caster = new THREE.Mesh(casterGeo, rimMat);
    caster.position.set(0.12, 0.015, 0);
    this.robotMeshGroup.add(caster);

    // 5. 360° Rotating 2D LiDAR Turret (mvsim_model/turtlebot3/lds.dae)
    const puckBaseGeo = new THREE.CylinderGeometry(0.045, 0.05, 0.03, 20);
    const puckBase = new THREE.Mesh(puckBaseGeo, plateMat);
    puckBase.position.y = 0.195;
    this.robotMeshGroup.add(puckBase);

    const puckHeadGeo = new THREE.CylinderGeometry(0.042, 0.042, 0.025, 20);
    const puckHeadMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, emissive: 0x06b6d4, emissiveIntensity: 0.35 });
    this.robotLidarPuck = new THREE.Mesh(puckHeadGeo, puckHeadMat);
    this.robotLidarPuck.position.y = 0.22;
    this.robotMeshGroup.add(this.robotLidarPuck);

    // 6. Forward Camera Mast (mvsim_model/turtlebot3/r200.dae)
    const camGeo = new THREE.BoxGeometry(0.025, 0.02, 0.09);
    const camMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7 });
    const cam = new THREE.Mesh(camGeo, camMat);
    cam.position.set(0.13, 0.14, 0);
    this.robotMeshGroup.add(cam);

    // Forward Headlight Beams
    const headlight = new THREE.SpotLight(0x06b6d4, 1.8, 8, Math.PI / 5, 0.4);
    headlight.position.set(0.15, 0.14, 0);
    headlight.target.position.set(3.0, 0.05, 0);
    this.robotMeshGroup.add(headlight);
    this.robotMeshGroup.add(headlight.target);

    this.scene.add(this.robotMeshGroup);
  }

  // =========================================================================
  // Mission Target Holographic Beacons (7 College Destinations)
  // =========================================================================
  buildTargetBeacons() {
    this.targetBeaconsGroup.clear();
    const targets = this.app.world.targets;

    targets.forEach((tgt, idx) => {
      const bGroup = new THREE.Group();
      bGroup.position.set(tgt.x, 0, -tgt.y);

      // Rotating holographic rings
      const ringGeo = new THREE.RingGeometry(0.45, 0.6, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.75
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.04;
      bGroup.add(ring);

      // Vertical beacon column
      const colGeo = new THREE.CylinderGeometry(0.04, 0.04, 2.5, 8);
      const colMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.6 });
      const col = new THREE.Mesh(colGeo, colMat);
      col.position.y = 1.25;
      bGroup.add(col);

      // Floating department label billboard
      const label = this.createBeaconLabel(`${idx + 1}. ${tgt.name}`);
      label.position.y = 2.8;
      bGroup.add(label);

      this.targetBeaconsGroup.add(bGroup);
    });
  }

  createBeaconLabel(text) {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.roundRect ? ctx.roundRect(5, 5, 374, 86, 16) : ctx.rect(5, 5, 374, 86);
    ctx.fill();

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 4;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 28px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 192, 48);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.8, 0.7, 1);
    return sprite;
  }

  // =========================================================================
  // Per-Frame 3D Update Loop
  // =========================================================================
  update(dt) {
    if (!this.enabled) return;

    const veh = this.app.world.vehicles.get(this.app.activeVehicleName);
    const scanner = this.app.sensors.get(this.app.activeVehicleName);
    const apf = this.app.apfs.get(this.app.activeVehicleName);

    if (veh && this.robotMeshGroup) {
      // 1. Update 3D Robot Position & Heading
      // In Three.js: X is East/West, Y is Up, Z is North/South (flipped Y in 2D space)
      this.robotMeshGroup.position.set(veh.pose.x, 0, -veh.pose.y);
      this.robotMeshGroup.rotation.y = veh.pose.yaw;

      // 2. Animate Drive Wheels rotation
      if (this.robotWheels.length >= 2) {
        const wVel = veh.twist.vx / 0.033; // v / r
        this.robotWheels[0].rotation.z += wVel * dt;
        this.robotWheels[1].rotation.z += wVel * dt;
      }

      // 3. Spin 360° LiDAR Puck
      if (this.robotLidarPuck) {
        this.robotLidarPuck.rotation.y += 18.0 * dt; // 10 Hz rotation
      }

      // 4. Update 3D Camera based on selected mode
      this.updateCamera(veh);

      // 5. Render 3D LiDAR point cloud & beams
      if (scanner) {
        this.updateLiDAR3D(veh, scanner);
      }

      // 6. Render 3D A* Planned Navigation Path
      if (apf) {
        this.updatePathRibbon3D(apf);
      }
    }

    // Render WebGL frame
    this.renderer.render(this.scene, this.camera);
  }

  updateCamera(veh) {
    if (this.cameraMode === 'follow') {
      // 3rd-person chase camera behind robot
      const camDist = 5.2;
      const camHeight = 3.6;
      const targetCamX = veh.pose.x - camDist * Math.cos(veh.pose.yaw);
      const targetCamZ = -veh.pose.y + camDist * Math.sin(veh.pose.yaw);

      this.camera.position.x += (targetCamX - this.camera.position.x) * 0.08;
      this.camera.position.y += (camHeight - this.camera.position.y) * 0.08;
      this.camera.position.z += (targetCamZ - this.camera.position.z) * 0.08;

      const lookTarget = new THREE.Vector3(
        veh.pose.x + 2.0 * Math.cos(veh.pose.yaw),
        0.8,
        -veh.pose.y - 2.0 * Math.sin(veh.pose.yaw)
      );
      this.camera.lookAt(lookTarget);

    } else if (this.cameraMode === 'fpv') {
      // 1st-person FPV Robot Camera
      const eyeHeight = 0.22;
      this.camera.position.set(veh.pose.x, eyeHeight, -veh.pose.y);
      const lookTarget = new THREE.Vector3(
        veh.pose.x + 8.0 * Math.cos(veh.pose.yaw),
        eyeHeight - 0.05,
        -veh.pose.y - 8.0 * Math.sin(veh.pose.yaw)
      );
      this.camera.lookAt(lookTarget);

    } else if (this.cameraMode === 'topdown') {
      // Drone Tactical Camera
      this.camera.position.set(veh.pose.x, 28.0, -veh.pose.y);
      this.camera.lookAt(veh.pose.x, 0, -veh.pose.y);

    } else if (this.orbitControls) {
      // Free Orbit Mode
      this.orbitControls.update();
    }
  }

  updateLiDAR3D(veh, scanner) {
    // 3D Point Cloud Particles
    const validHits = scanner.hitPointsWorld;
    if (!validHits || validHits.length === 0) return;

    if (!this.lidarPointsMesh) {
      const geo = new THREE.BufferGeometry();
      const posArray = new Float32Array(361 * 3);
      geo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      const mat = new THREE.PointsMaterial({ color: 0x06b6d4, size: 0.14, transparent: true, opacity: 0.9 });
      this.lidarPointsMesh = new THREE.Points(geo, mat);
      this.scene.add(this.lidarPointsMesh);
    }

    const pos = this.lidarPointsMesh.geometry.attributes.position.array;
    let pIdx = 0;
    for (let i = 0; i < validHits.length; i++) {
      pos[pIdx++] = validHits[i].x;
      pos[pIdx++] = 0.22; // LiDAR height
      pos[pIdx++] = -validHits[i].y;
    }
    this.lidarPointsMesh.geometry.setDrawRange(0, validHits.length);
    this.lidarPointsMesh.geometry.attributes.position.needsUpdate = true;
  }

  updatePathRibbon3D(apf) {
    if (!apf.activePath || apf.activePath.length < 2) {
      if (this.pathRibbonMesh) this.pathRibbonMesh.visible = false;
      return;
    }

    const pts = apf.activePath.map(p => new THREE.Vector3(p[0], 0.08, -p[1]));
    const curve = new THREE.CatmullRomCurve3(pts);
    const tubeGeo = new THREE.TubeGeometry(curve, Math.max(12, pts.length * 4), 0.06, 8, false);

    if (!this.pathRibbonMesh) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x10b981, wireframe: false });
      this.pathRibbonMesh = new THREE.Mesh(tubeGeo, mat);
      this.scene.add(this.pathRibbonMesh);
    } else {
      this.pathRibbonMesh.geometry.dispose();
      this.pathRibbonMesh.geometry = tubeGeo;
      this.pathRibbonMesh.visible = true;
    }
  }

  setupEventListeners() {
    window.addEventListener('resize', () => {
      if (!this.container) return;
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });

    // 3D Click to Add Custom Target or Drag Obstacles
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => {
      const rect = dom.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObjects(this.interactiveObstaclesGroup.children, true);

      if (intersects.length > 0) {
        this.isDragging = true;
        this.draggedMesh = intersects[0].object.parent || intersects[0].object;
        if (this.orbitControls) this.orbitControls.enabled = false;
      }
    });

    dom.addEventListener('pointermove', (e) => {
      if (!this.isDragging || !this.draggedMesh) return;
      const rect = dom.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersectPoint = new THREE.Vector3();
      this.raycaster.ray.intersectPlane(this.dragPlane, intersectPoint);

      if (intersectPoint) {
        this.draggedMesh.position.x = intersectPoint.x;
        this.draggedMesh.position.z = intersectPoint.z;
      }
    });

    const stopDrag = () => {
      this.isDragging = false;
      this.draggedMesh = null;
      if (this.orbitControls) this.orbitControls.enabled = (this.cameraMode === 'orbit');
    };
    window.addEventListener('pointerup', stopDrag);
    window.addEventListener('pointercancel', stopDrag);
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    if (this.orbitControls) {
      this.orbitControls.enabled = (mode === 'orbit');
    }
  }
}

window.MVSim3DEngine = MVSim3DEngine;

