import os
import torch
import numpy as np
import open3d as o3d
from flask import Flask, render_template, jsonify
import json
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.camera_utils import cameraObjectsNoImage_from_cameraInfos
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dataclasses import field

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    data_path: str = "data"
    dataset_name: str = "dataset1"
    sequence_name: str = "sequence0001"
    checkpoint_path: str = field(init=False)
    data_device: str = "cuda"

    def __post_init__(self):
        self.checkpoint_path = os.path.join(
            self.data_path,
            self.dataset_name,
            self.sequence_name,
            "gm_checkpoints",
            "static_gaussian.pth"
        )

def load_gaussian_model(config: VisualizationConfig) -> Dict[str, Any]:
    """Load the Gaussian model from checkpoint."""
    # Initialize model
    gaussians = GaussianModel(sh_degree=3)
    
    # Load checkpoint
    model_params = torch.load(config.checkpoint_path)
    gaussians.restore(model_params, config)
    
    # Convert to numpy for visualization
    xyz = gaussians._xyz.detach().cpu().numpy()
    rgb = gaussians._features_dc.detach().cpu().numpy()
    opacity = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()
    
    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation
    }

def create_point_cloud(model_data: Dict[str, Any]) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud from Gaussian model data."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(model_data["xyz"])
    pcd.colors = o3d.utility.Vector3dVector(model_data["rgb"])
    
    # Add custom attributes for visualization
    pcd.point = {
        "opacity": model_data["opacity"],
        "scale": model_data["scale"],
        "rotation": model_data["rotation"]
    }
    
    return pcd

def create_web_viewer(model_data: Dict[str, Any], output_dir: str = "static"):
    """Create web-based visualization using Flask and Three.js."""
    app = Flask(__name__)
    
    # Create output directory for static files
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model data as JSON for web viewer
    model_json = {
        "points": model_data["xyz"].tolist(),
        "colors": model_data["rgb"].tolist(),
        "opacity": model_data["opacity"].tolist(),
        "scale": model_data["scale"].tolist(),
        "rotation": model_data["rotation"].tolist()
    }
    
    with open(os.path.join(output_dir, "model_data.json"), "w") as f:
        json.dump(model_json, f)
    
    @app.route("/")
    def index():
        return render_template("viewer.html")
    
    @app.route("/model_data")
    def get_model_data():
        return jsonify(model_json)
    
    return app

def main():
    # Initialize configuration
    config = VisualizationConfig()
    
    # Load model data
    print(f"Loading model from {config.checkpoint_path}")
    model_data = load_gaussian_model(config)
    
    # Create Open3D visualization
    print("Creating Open3D visualization...")
    pcd = create_point_cloud(model_data)
    
    # Create web viewer
    print("Creating web viewer...")
    app = create_web_viewer(model_data)
    
    # Start web server
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser to view the model")
    app.run(debug=True)

if __name__ == "__main__":
    main() 