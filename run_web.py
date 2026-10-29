"""
MVSim 3D College Campus Simulation Web Server.
Serves the 3D WebGL simulator at '/' and serves models from 'mvsim_model/' at '/mvsim_model/'.

Usage:
  python run_web.py
  python run_web.py --port 8085 --no-browser
"""

import os
import sys
import argparse
import webbrowser
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Add MIME types for 3D assets
mimetypes.add_type("model/vnd.collada+xml", ".dae")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/octet-stream", ".bin")


class MVSim3DHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mvsim_extracted", "web")
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mvsim_model")
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        # Route /mvsim_model/... directly to D:/SImulation_bot/mvsim_model/...
        clean_path = path.split("?", 1)[0].split("#", 1)[0]
        if clean_path.startswith("/mvsim_model/"):
            rel = clean_path[len("/mvsim_model/"):]
            return os.path.join(self.models_dir, *rel.split("/"))
        # Default route to web/
        rel = clean_path.lstrip("/")
        return os.path.join(self.web_dir, *rel.split("/"))

    def end_headers(self):
        # Enable CORS for WebGL textures and assets
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="MVSim 3D Simulation Web Server")
    parser.add_argument("--port", type=int, default=8085, help="Port to bind (default: 8085)")
    parser.add_argument("--host", type=str, default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, MVSim3DHandler)

    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 65)
    print("MVSim 3D College Campus Simulator")
    print(f"Server running at: {url}")
    print("=" * 65)
    print("Features:")
    print("  - Full 3D College Campus with Department Buildings & Landscaping")
    print("  - TurtleBot3 Burger Robot (mvsim_model/turtlebot3) with spinning wheels & LiDAR")
    print("  - 3D Model Props: Bookshelves, Tool Carts, Dining Sets, Piano, Radio Tower")
    print("  - 3D Real-time LiDAR point cloud & A* planned trajectory ribbon")
    print("  - Multi-camera: 3rd-person Follow, 1st-person FPV, Free Orbit, Top-Down")
    print("  - Real-time drag-and-drop obstacle manipulation in 3D")
    print("=" * 65 + "\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down 3D simulation server.")
        httpd.server_close()


if __name__ == "__main__":
    main()

