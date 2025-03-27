# 1_static_preprocessing.py
# This script trains a single static gaussian model for t=0 with densification

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Tuple, Optional
import os
import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from random import randint
from torch.utils.data import Dataset
import json
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# local imports
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer_gsplat import render

@dataclass
class ExperimentConfig:
    """Configuration for experiment setup and data paths."""
    data_path: str = "data"
    dataset_name: str = "dataset1"
    dataset_path: str = field(init=False)
    gm_output_path: str = field(init=False)
    data_device: str = "cuda"

    def __post_init__(self):
        self.dataset_path = f"{self.data_path}/{self.dataset_name}"
        self.gm_output_path = self.dataset_path

@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    exposure_lr_init: float = 0.01
    exposure_lr_final: float = 0.001
    exposure_lr_delay_steps: int = 0
    exposure_lr_delay_mult: float = 0.0
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.01
    random_background: bool = False
    optimizer_type: str = "default"
    semantic_feature_lr: float = 0.001  # Learning rate for semantic features
    noise_scale: float = 1.0  # Scaling factor for annealing noise (Nε)
    noise_end_iter: int = 10000  # Iteration at which noise reaches zero

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    sh_degree: int = 3
    model_path: str = ""
    resolution: int = -1
    white_background: bool = False
    eval: bool = False
    speedup: bool = False
    render_items: List[str] = field(default_factory=lambda: ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"])
    manual_gaussians_bool: bool = False

@dataclass
class PipelineConfig:
    """Configuration for pipeline parameters."""
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = True

def flatten_dataclass(cls):
    """Decorator to flatten nested dataclass attributes while preserving original structure."""
    original_init = cls.__init__
    
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for field in fields(cls):
            if hasattr(field.type, '__dataclass_fields__'):
                nested_obj = getattr(self, field.name)
                for nested_field in fields(nested_obj):
                    if not hasattr(cls, nested_field.name):
                        def make_property(field_name, nested_field_name):
                            return property(
                                lambda self, field_name=field_name, nested_field_name=nested_field_name: 
                                getattr(getattr(self, field_name), nested_field_name)
                            )
                        setattr(cls, nested_field.name, make_property(field.name, nested_field.name))
                        # Also set the attribute directly for easier access
                        setattr(cls, nested_field.name, getattr(nested_obj, nested_field.name))
    
    cls.__init__ = __init__
    return cls

@flatten_dataclass
@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    experiment: ExperimentConfig = ExperimentConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    
    # Add direct attributes for commonly accessed nested values
    data_device: str = field(init=False)
    dataset_path: str = field(init=False)
    gm_output_path: str = field(init=False)
    
    def __post_init__(self):
        # Set direct attributes from nested experiment config
        self.data_device = self.experiment.data_device
        self.dataset_path = self.experiment.dataset_path
        self.gm_output_path = self.experiment.gm_output_path


def select_random_camera(scene: Scene) -> Any:
    """Select a random camera from the scene's camera stack."""
    cam_stack = scene.getTrainCameraObjects()
    return cam_stack[randint(0, len(cam_stack) - 1)]

def visualize_gaussians(
    scene: Scene,
    gaussians: GaussianModel,
    config: Config,
    background: torch.Tensor,
    cam_stack: List[Any],
    name: Optional[str] = None,
    gt_images: Optional[torch.Tensor] = None
) -> None:
    """Visualize gaussians from multiple camera viewpoints."""
    for cam_idx, camera in enumerate(cam_stack):
        render_pkg = render(camera, gaussians, config.pipeline, bg_color=background)
        rendered_image = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        
        # Use the loaded ground truth image if available
        if gt_images is not None:
            gt_image = gt_images[cam_idx].cpu().numpy().transpose(1, 2, 0)
        else:
            # If no ground truth available, create a black image
            gt_image = np.zeros_like(rendered_image)
        
        # Create white separator line
        H, W, C = rendered_image.shape
        white_line = np.full((H, 2, C), 255, dtype=np.uint8)
        
        # Concatenate horizontally: [rendered | white_line | gt]
        combined_image = np.concatenate([rendered_image, white_line, gt_image], axis=1)
        combined_image = (combined_image * 255).astype(np.uint8)

        wandb.log({f"{name}_cam{cam_idx}": wandb.Image(combined_image)})


def densification_step(
    iteration: int,
    opt: OptimizationConfig,
    gaussians: GaussianModel,
    render_pkg: Dict[str, torch.Tensor],
    visibility_filter: torch.Tensor,
    radii: torch.Tensor,
    viewspace_point_tensor: torch.Tensor,
    scene: Scene,
    dataset: ModelConfig
) -> None:
    """Perform densification and pruning steps during training."""
    if iteration < opt.densify_until_iter:
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        gaussians.densify_and_prune(
            opt.densify_grad_threshold,
            0.005,
            scene.cameras_extent,
            size_threshold
        )
    if iteration % opt.opacity_reset_interval == 0 or (
        dataset.white_background and iteration == opt.densify_from_iter
    ):
        gaussians.reset_opacity()


def log_gaussian_distributions(gaussians: GaussianModel, iteration: int, loss: float) -> None:
    """Log distributions of gaussian parameters to wandb."""
    # Compute opacity and scale distributions
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    
    # Create histograms
    opacity_hist, opacity_bins = np.histogram(opacities, bins=50, range=(0, 1))
    scale_hist, scale_bins = np.histogram(scales, bins=50)
    
    # Compute CDFs
    opacity_cdf = np.cumsum(opacity_hist) / len(opacities)
    scale_cdf = np.cumsum(scale_hist) / len(scales)
    
    # Log distributions to wandb
    wandb.log({
        "loss": loss.item(),
        "num_gaussians": len(gaussians._xyz),
        "iteration": iteration,
        "opacity_histogram": wandb.Histogram(np_histogram=(opacity_hist, opacity_bins)),
        "scale_histogram": wandb.Histogram(np_histogram=(scale_hist, scale_bins)),
        "opacity_cdf": wandb.plot.line_series(
            xs=[opacity_bins[:-1].tolist()],
            ys=[opacity_cdf.tolist()],
            keys=["CDF"],
            title="Opacity CDF"
        ),
        "scale_cdf": wandb.plot.line_series(
            xs=[scale_bins[:-1].tolist()],
            ys=[scale_cdf.tolist()],
            keys=["CDF"],
            title="Scale CDF"
        ),
        "opacity_stats": {
            "mean": float(np.mean(opacities)),
            "std": float(np.std(opacities)),
            "min": float(np.min(opacities)),
            "max": float(np.max(opacities)),
            "median": float(np.median(opacities))
        },
        "scale_stats": {
            "mean": float(np.mean(scales)),
            "std": float(np.std(scales)),
            "min": float(np.min(scales)),
            "max": float(np.max(scales)),
            "median": float(np.median(scales))
        }
    })

def apply_annealing_noise(xyz: torch.Tensor, iteration: int, config: OptimizationConfig) -> torch.Tensor:
    """Apply annealing noise to the gaussian positions.
    
    Args:
        xyz: Tensor of gaussian positions (N, 3)
        iteration: Current training iteration
        config: Optimization config containing noise parameters
        
    Returns:
        Tensor of noised positions (N, 3)
    """
    # Compute noise scale that decreases linearly until noise_end_iter
    noise_factor = 1.0 - min(1.0, iteration / config.noise_end_iter)
    
    # Generate random noise
    noise = torch.randn_like(xyz, device=xyz.device) * config.noise_scale * noise_factor
    
    return xyz + noise

def train_static_gaussian_model(
    scene: Scene,
    config: Config,
    iterations: int = 30000
) -> None:
    """Train a single static gaussian model with densification."""
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    scene.initialize_gaussians_from_scene_info(scene.gaussians, config.model)
    scene.gaussians.training_setup_0(config.optimization)

    # Load ground truth images from first frame of videos
    scene_dir = scene.dataset_path
    dynamic_dir = os.path.join(scene_dir, "dynamic")
    video_files = sorted([f for f in os.listdir(dynamic_dir) if f.endswith(".mp4")])
    
    gt_images = []
    for video_file in video_files:
        video_path = os.path.join(dynamic_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()  # Read first frame
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Could not read first frame from {video_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_tensor = T.ToTensor()(frame)
        gt_images.append(frame_tensor)
    
    gt_images = torch.stack(gt_images).to("cuda")

    progress_bar = tqdm(range(iterations), desc="Training Static Gaussian Model")
    for iteration in progress_bar:
        viewpoint_cam = select_random_camera(scene)
        cam_idx = viewpoint_cam.uid
        
        # Apply annealing noise to gaussian positions
        noised_xyz = apply_annealing_noise(scene.gaussians._xyz, iteration, config.optimization)
        original_xyz = scene.gaussians._xyz.clone()
        scene.gaussians._xyz.data.copy_(noised_xyz)
        
        # Render
        render_pkg = render(viewpoint_cam, scene.gaussians, config.pipeline, bg_color)
        rendered_image = render_pkg["render"]
        
        # Compute loss with ground truth
        gt_image = gt_images[cam_idx]
        loss = F.mse_loss(rendered_image, gt_image)
        
        # Log basic metrics every iteration
        wandb.log({
            "loss": loss.item(),
            "num_gaussians": len(scene.gaussians._xyz),
            "iteration": iteration,
            "noise_scale": config.optimization.noise_scale * (1.0 - min(1.0, iteration / config.optimization.noise_end_iter))
        })
        
        # Log detailed distributions every 1000th iteration
        if iteration % 1000 == 0:
            log_gaussian_distributions(scene.gaussians, iteration, loss)
        
        loss.backward()
        
        # Restore original positions before optimizer step
        scene.gaussians._xyz.data.copy_(original_xyz)

        scene.gaussians.update_learning_rate(iteration)
        scene.gaussians.optimizer.step()
        scene.gaussians.optimizer.zero_grad(set_to_none=True)

        # Densification step
        if iteration % 500 == 0:
            densification_step(
                iteration,
                config.optimization,
                scene.gaussians,
                render_pkg,
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["viewspace_points"],
                scene,
                config.model
            )
            
        # Visualization
        if iteration % 5000 == 0:
            visualize_gaussians(
                scene,
                scene.gaussians,
                config,
                bg_color,
                cam_stack=scene.getTrainCameraObjects()[:1],
                name=f"iteration_{iteration}",
                gt_images=gt_images
            )

        progress_bar.set_postfix({
            "Loss": f"{loss.item():.7f}",
            "Gaussians": f"{len(scene.gaussians._xyz)}"
        })
        progress_bar.update(1)
    
    # Final visualization
    visualize_gaussians(
        scene,
        scene.gaussians,
        config,
        bg_color,
        cam_stack=scene.getTrainCameraObjects()[:1],
        name=f"final_{scene.dataset_path.split('/')[-1]}_t0",
        gt_images=gt_images
    )

def train(config: Config) -> None:
    """Main training function for static gaussian model."""
    device = config.experiment.data_device
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    dataset = PreprocessingDataset(config)
    print(f"Found {len(dataset)} scenes for training.")
    
    for seq_idx in range(len(dataset)):
        print(f"\nProcessing scene: {os.path.basename(dataset.scene_dirs[seq_idx])}")
        
        # Convert camera data
        sample = dataset[(seq_idx, 0)]
        cameras_gt_path = os.path.join(sample['scene_dir'], 'cameras_gt.json')
        train_meta_path = os.path.join(sample['scene_dir'], "train_meta.json")
        
        with open(cameras_gt_path, 'r') as f:
            cameras_gt_json = json.load(f)
        train_meta_json = convert_cameras_gt_to_train_meta(config, sample['scene_dir'], cameras_gt_json)
        with open(train_meta_path, 'w') as f:
            json.dump(train_meta_json, f)
        print(f"Saved train_meta.json to {train_meta_path}")

        # Train static gaussian model
        scene = Scene(config=config, scene_path=sample['scene_dir'])
        scene.gaussians = GaussianModel(sh_degree=config.model.sh_degree)
        
        checkpoint_path = os.path.join(sample['scene_dir'], "gm_checkpoints", "static_gaussian.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        train_static_gaussian_model(scene, config, iterations=config.optimization.iterations)
        torch.save(scene.gaussians.capture(), checkpoint_path)
        print(f"Saved static gaussian model with {len(scene.gaussians._xyz)} gaussians to {checkpoint_path}")

class PreprocessingDataset(Dataset):
    """Dataset class for preprocessing and loading scene data."""
    def __init__(self, config: Config):
        self.config = config
        self.dataset_path = config.experiment.dataset_path
        self.to_tensor = T.ToTensor()

        # Find sequence directories in train and test subdirectories
        train_path = os.path.join(self.dataset_path, "train")
        test_path = os.path.join(self.dataset_path, "test")
        
        train_sequences = sorted([
            os.path.join(train_path, d)
            for d in os.listdir(train_path)
            if os.path.isdir(os.path.join(train_path, d)) and d.startswith("sequence")
        ])
        
        test_sequences = sorted([
            os.path.join(test_path, d)
            for d in os.listdir(test_path)
            if os.path.isdir(os.path.join(test_path, d)) and d.startswith("sequence")
        ])
        
        self.scene_dirs = train_sequences + test_sequences

        if not self.scene_dirs:
            raise ValueError(
                f"No scene directories found in {train_path} or {test_path}. "
                "Make sure your data folder has subdirectories named like 'sequence0001', 'sequence0002', etc. "
                "under both train/ and test/ directories."
            )

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, idx_tuple: Tuple[int, int]) -> Dict[str, Any]:
        """Get data for a specific sequence and timestep."""
        sequence_idx, time_index = idx_tuple
        scene_dir = self.scene_dirs[sequence_idx]
        dynamic_dir = os.path.join(scene_dir, "dynamic")
        video_files = sorted([f for f in os.listdir(dynamic_dir) if f.endswith(".mp4")])

        frames = []
        for video_file in video_files:
            video_path = os.path.join(dynamic_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, time_index)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError(f"Could not read frame {time_index} from {video_path}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame_tensor = self.to_tensor(frame)
            frames.append(frame_tensor)

        return {
            "scene_dir": scene_dir,
            "gt_images": torch.stack(frames)
        }

def convert_cameras_gt_to_train_meta(config, scene_path, cameras_gt_json):
    """Convert cameras_gt.json to train_meta.json"""
    data = {}
    data['width'] = cameras_gt_json[0]['width']
    data['height'] = cameras_gt_json[0]['height']
    
    w2c, c2w, k, cam_id, img_path, vid_path = [], [], [], [], [], []

    for entry in cameras_gt_json:
        cam_idx = entry['camera_id']
        static_img_path = os.path.join(scene_path, "static", entry['img_fn'])
        img_path.append(static_img_path)
        vid_path.append(os.path.join(scene_path, "dynamic", entry['vid_fn']))
        w2c.append(entry['w2c'])
        c2w.append(entry['c2w'])
        k.append(get_intrinsics(entry))
        cam_id.append(str(cam_idx))

    data['w2c'] = w2c
    data['c2w'] = c2w
    data['k'] = k
    data['cam_id'] = cam_id
    data['img_path'] = img_path
    data['vid_path'] = vid_path

    return data

def get_intrinsics(camera_info):
    """Get intrinsics matrix from camera info"""
    fx = camera_info['fx']
    fy = camera_info['fy']
    cx = camera_info['width'] / 2
    cy = camera_info['height'] / 2
    k = [[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]
    return k

if __name__ == "__main__":
    config = Config()
    wandb.init(project="blender_static_preprocessing_debug")
    train(config)
    wandb.finish()
