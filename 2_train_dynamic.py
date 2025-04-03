import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
from torchdiffeq import odeint
import numpy as np
import cv2
import os
from tqdm import tqdm
import wandb
from collections import OrderedDict
from typing import Tuple, Dict, List, Optional, Any

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer_gsplat import render_batch


# --------------------------
# Configuration
# --------------------------
class Config:
    """Centralized configuration for the training setup."""
    def __init__(self):
        # Path configurations
        self.project_dir = "/users/williamb/dev/cvpr_gaussian_node"   #"/work/williamb"
        self.data_path = f"{self.project_dir}/data"
        self.dataset_name = "dataset1"
        self.dataset_path = f"{self.data_path}/{self.dataset_name}"
        
        # Device settings
        self.data_device = "cuda"
        self.device = "cuda"
        
        # Training parameters
        self.framerate = 25  # Samples per second
        self.wandb_fps = 5
        self.initial_segment_length_seconds = 0.2
        self.num_timestep_samples = 5
        self.num_train_sequences = 1
        self.num_test_sequences = 1
        self.epochs = 300_000
        self.batch_size = 1
        self.initial_segment_length = 30
        self.photometric_loss_length = 30
        self.test_loss_length = 30
        self.learning_rate = 5e-3
        self.loss_threshold = 1.2e-4
        
        # Decaying parameters learning rate configuration
        self.decaying_params_lr_scale = 1.0  # Initial scale factor for decaying parameters
        self.decaying_params_lr_decay = 0.999  # Exponential decay factor per epoch
        
        # Gradient clipping and regularization
        self.grad_clip_value = 0.2
        # self.reg_weight = 0.001
        
        # ODE solver configuration
        self.ode_method = 'dopri5'  # Use adaptive step size solver
        self.ode_rtol = 1e-3
        self.ode_atol = 1e-4
        
        # Visualization
        self.train_visualization_interval = 250
        self.test_visualization_interval = 100
        self.test_loss_interval = 100
        
        # Gaussian model pipeline
        self.pipeline = {
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "debug": True
        }
        
        # Rigidity
        self.rigidity_sample_size = 100
        self.rigidity_neighbors = 3
        
        # Annealing noise parameters
        self.noise_scale = 1  # Scaling factor Nε for annealing noise
        self.noise_end_iter = 2000  # Iteration at which noise reaches zero
        
        # Logging
        self.use_wandb = True  # Flag to control wandb usage
        self.wandb_name = "latent_node"

# --------------------------
# Math Utilities
# --------------------------
class QuaternionOps:
    """Static class for quaternion operations."""
    @staticmethod
    def inverse(q: torch.Tensor) -> torch.Tensor:
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    @staticmethod
    def to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(1)
        N = quat.shape[0]
        
        R = torch.zeros((N, 3, 3), device=quat.device)
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        R[:, 0, 0] = ww + xx - yy - zz
        R[:, 0, 1] = 2*(xy - wz)
        R[:, 0, 2] = 2*(xz + wy)
        R[:, 1, 0] = 2*(xy + wz)
        R[:, 1, 1] = ww - xx + yy - zz
        R[:, 1, 2] = 2*(yz - wx)
        R[:, 2, 0] = 2*(xz - wy)
        R[:, 2, 1] = 2*(yz + wx)
        R[:, 2, 2] = ww - xx - yy + zz
        
        return R

    @staticmethod
    def exp(omega: torch.Tensor) -> torch.Tensor:
        """Compute quaternion exponential map.
        
        Args:
            omega: Angular velocity tensor of shape [..., 3]
            
        Returns:
            Quaternion tensor of shape [..., 4]
        """
        # Compute angle (magnitude of angular velocity)
        theta = torch.norm(omega, dim=-1, keepdim=True)
        
        # Handle small angles (Taylor series approximation)
        small_angle_mask = theta < 1e-6
        theta = torch.where(small_angle_mask, torch.ones_like(theta), theta)
        
        # Compute sin(theta/2) and cos(theta/2)
        sin_half = torch.sin(0.5 * theta)
        cos_half = torch.cos(0.5 * theta)
        
        # Normalize angular velocity for direction
        omega_norm = omega / (theta + 1e-8)
        
        # Construct quaternion [cos(theta/2), sin(theta/2) * omega_norm]
        quat = torch.cat([
            cos_half,
            sin_half * omega_norm[..., 0:1],
            sin_half * omega_norm[..., 1:2],
            sin_half * omega_norm[..., 2:3],
        ], dim=-1)
        
        # For very small angles, use first-order approximation
        quat = torch.where(
            small_angle_mask.expand_as(quat),
            torch.cat([
                torch.ones_like(cos_half),
                0.5 * omega[..., 0:1],
                0.5 * omega[..., 1:2],
                0.5 * omega[..., 2:3]
            ], dim=-1),
            quat
        )
        
        return quat

    @staticmethod
    def log(q: torch.Tensor) -> torch.Tensor:
        """Compute quaternion logarithm map.
        
        Args:
            q: Quaternion tensor of shape [..., 4]
            
        Returns:
            Angular velocity tensor of shape [..., 3]
        """
        # Extract scalar and vector parts
        w = q[..., 0:1]
        v = q[..., 1:]
        
        # Compute angle
        theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
        
        # Handle small angles
        small_angle_mask = theta < 1e-6
        theta = torch.where(small_angle_mask, torch.ones_like(theta), theta)
        
        # Compute normalized vector part
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_normalized = v / (v_norm + 1e-8)
        
        # Compute logarithm
        omega = theta * v_normalized
        
        # For very small angles, use first-order approximation
        omega = torch.where(
            small_angle_mask.expand_as(omega),
            2 * v,
            omega
        )
        
        return omega

# --------------------------
# Neural Network Components
# --------------------------
class MLP(nn.Module):
    """Basic Multi-Layer Perceptron."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = []
        dim = input_dim
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.Tanh()])
            dim = hidden_dim
            
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualBlock(nn.Module):
    """Residual block with optional activation."""
    def __init__(self, in_features: int, out_features: int, is_final: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else nn.Identity()
        self.activation = None if is_final else nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x) + self.shortcut(x)
        return self.activation(out) if self.activation else out

class ResidualMLP(nn.Module):
    """Residual MLP network."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i+1], i == len(dims)-2)
            for i in range(len(dims)-1)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

# --------------------------
# ODE Components
# --------------------------
class ODEFunc(nn.Module):
    """Neural ODE function."""
    def __init__(self, latent_dim: int = 128, num_layers: int = 4, hidden_dim: int = 256, config: Config = None):
        super().__init__()
        self.nfe = 0
        
        # Single unified network for both velocity and angular velocity
        self.dynamics_net = ResidualMLP(13 + latent_dim + 1, 6, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # Regularization terms
        self.reg_loss = 0.0
        self.iteration = 0
        self.config = config
        
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        z: [batch_size, num_points, 13+latent_dim] with xyz(0:3), quat(3:7), vel(7:10), omega(10:13), latent(13:)
        Compute the derivative dz/dt.
        """
        self.nfe += 1
        batch_size, num_points, dim = z.shape
        
        # Normalize quaternions
        quat = z[..., 3:7].clone()
        # quat = F.normalize(quat, p=2, dim=-1)
        z = torch.cat([z[..., :3], quat, z[..., 7:]], dim=-1)
        # z = apply_annealing_noise(z, self.iteration, self.config)
        
        # Compute derivatives
        t_expanded = t.view(-1, 1, 1).expand(batch_size, num_points, 1)
        zt = torch.cat([z, t_expanded], dim=-1)
        
        # Position derivative is velocity
        dt_pos = z[..., 7:10]  # velocity
        
        # Unified prediction of velocity and angular velocity derivatives
        combined_dynamics = self.dynamics_net(zt)
        dt_vel = combined_dynamics[..., :3]
        dt_omega = combined_dynamics[..., 3:6]
        
        # Add sparsity and smoothness regularization
        vel_norm = torch.norm(dt_vel, dim=-1).mean()
        omega_norm = torch.norm(dt_omega, dim=-1).mean()
        # self.reg_loss = 0.01 * (vel_norm + omega_norm)  # L1 regularization to promote sparsity
        
        # Quaternion derivative using exponential map
        omega = z[..., 10:13]
        dt_quat = QuaternionOps.exp(omega)  # This gives us the quaternion derivative
        
        # Latent derivative is zero
        dt_latent = torch.zeros_like(z[..., 13:])
        
        return torch.cat([dt_pos, dt_quat, dt_vel, dt_omega, dt_latent], dim=-1)

# --------------------------
# Model Architecture
# --------------------------
class LatentNeuralODE(nn.Module):
    """Latent Neural ODE model with trainable initial conditions."""
    def __init__(self, num_points: int, latent_dim: int = 64, 
                    atol: float = 1e-5, rtol: float = 1e-3, method: str = 'dopri5', config: Config = None):
        super().__init__()
        self.func = ODEFunc(latent_dim=latent_dim, config=config)
        self.encoder = MLP(13, latent_dim)
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.config = config
        # Trainable initial conditions for each point
        self.initial_position_offset = nn.Parameter(torch.zeros(num_points, 3))
        self.initial_rotation_offset = nn.Parameter(torch.zeros(num_points, 4))
        self.initial_velocity = nn.Parameter(torch.zeros(num_points, 3))
        self.initial_omega = nn.Parameter(torch.zeros(num_points, 3))
        
    def forward(self, z0: torch.Tensor, t_span: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        """Forward pass through the ODE solver.
        
        Args:
            z0: Initial state tensor of shape [batch_size, num_points, 6] containing position and rotation
            t_span: Time points tensor of shape [num_timesteps]
            
        Returns:
            Trajectory tensor of shape [num_timesteps, batch_size, num_points, 13]
        """
        # Combine initial state with trainable velocities
        pos0_delta = self.initial_position_offset.unsqueeze(0)
        rot0_delta = self.initial_rotation_offset.unsqueeze(0)
        z0_delta = torch.cat([pos0_delta, rot0_delta], dim=-1)
        z0 = z0 + z0_delta
        v0 = self.initial_velocity.unsqueeze(0)
        w0 = self.initial_omega.unsqueeze(0)
        # v0 = torch.zeros_like(self.initial_velocity.unsqueeze(0))
        # w0 = torch.zeros_like(self.initial_omega.unsqueeze(0))
        z0 = torch.cat([
            z0,  # Initial state [batch_size, num_points, 7]
            v0,  # Trainable initial velocity [1, num_points, 3]
            w0,  # Trainable initial angular velocity [1, num_points, 3]
        ], dim=-1)

        
        # Add latent encoding
        z0_encoded = self.encoder(z0)
        z0 = torch.cat([z0, z0_encoded], dim=-1)
        # z0 = apply_annealing_noise(z0, iteration, self.config)
        
        z_traj = odeint(self.func, z0, t_span, 
                        method=self.method, atol=self.atol, rtol=self.rtol,
                        options={'max_num_steps': 1000})  # Add explicit maximum steps for adaptive methods
        self.func.iteration += 1
        return z_traj # xyz(0:3), quat(3:7), vel(7:10), omega(10:13), latent(13:)

# --------------------------
# Data Loading
# --------------------------
class FrameCache:
    """Efficient frame caching system."""
    def __init__(self, max_size: int = 500):
        self.cache = OrderedDict()
        self.max_size = max_size
        
    def get(self, key: Tuple[int, int]) -> Optional[torch.Tensor]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: Tuple[int, int], value: torch.Tensor):
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

class SceneData:
    """Handles loading and managing scene data."""
    def __init__(self, config: Config, sequence_dir: str):
        self.config = config
        self.sequence_dir = sequence_dir
        self.frame_cache = FrameCache()
        self.scene = None
        self.gm0 = None
        self.gt_image = None
        self.gt_video = None
        self.z0 = None
        
        self._initialize_data()
        
    def _initialize_data(self):
        """Initialize scene data and Gaussian model."""
        # Load scene and Gaussian model
        self.scene = Scene(config=self.config, scene_path=self.sequence_dir)
        
        gm0_path = os.path.join(self.sequence_dir, "gm_checkpoints", "static_gaussian.pth")
        
        if not os.path.exists(gm0_path):
            raise FileNotFoundError(f"Missing static Gaussian model checkpoint in {self.sequence_dir}")
            
        self.gm0 = GaussianModel(sh_degree=3)
        self.scene.load_gaussians_from_checkpoint_for_msgnode(gm0_path, self.gm0)
        
        # Initialize data structures
        self.gt_image = self._load_gt_frames(0)
        
    def _load_gt_frames(self, timestep: int) -> torch.Tensor:
        """Load ground truth frames for a specific timestep."""
        frames = []
        dynamic_dir = os.path.join(self.sequence_dir, "dynamic")
        
        if os.path.exists(dynamic_dir):
            for video_file in sorted(f for f in os.listdir(dynamic_dir) if f.endswith('.mp4')):
                video_path = os.path.join(dynamic_dir, video_file)
                frame = self._load_single_frame(video_path, timestep)
                if frame is not None:
                    frames.append(frame)
                    
        return torch.stack(frames) if frames else None
        
    def _load_single_frame(self, video_path: str, timestep: int, for_training: bool = True) -> Optional[torch.Tensor]:
        """Load a single frame from video.
        
        Args:
            video_path: Path to the video file
            timestep: Frame index to load
            for_training: If True, return float32 tensor in [0, 1] range. If False, return uint8 tensor in [0, 255] range.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, timestep)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            if for_training:
                return frame.float() / 255.0
            return frame
        return None
        
    def get_frame(self, timestep: int, cam_idx: int, for_training: bool = True) -> Optional[torch.Tensor]:
        """Get a frame with caching support.
        
        Args:
            timestep: Frame index to load
            cam_idx: Camera index or slice
            for_training: If True, return float32 tensor in [0, 1] range. If False, return uint8 tensor in [0, 255] range.
        """
        if isinstance(cam_idx, slice):
            # Handle slice input by getting frames for all cameras in the slice
            frames = []
            for idx in range(cam_idx.start, cam_idx.stop):
                cache_key = (idx, timestep, for_training)
                
                # Check cache first
                cached_frame = self.frame_cache.get(cache_key)
                if cached_frame is not None:
                    frames.append(cached_frame)
                    continue
                    
                # Check if in preloaded video
                if self.gt_video is not None and timestep < self.gt_video.shape[0]:
                    frame = self.gt_video[timestep, idx]
                    if for_training:
                        frame = frame.float() / 255.0
                    frames.append(frame)
                    continue
                    
                # Load from disk
                dynamic_dir = os.path.join(self.sequence_dir, "dynamic")
                if not os.path.exists(dynamic_dir):
                    continue
                    
                video_files = sorted(f for f in os.listdir(dynamic_dir) if f.endswith('.mp4'))
                if idx >= len(video_files):
                    continue
                    
                video_path = os.path.join(dynamic_dir, video_files[idx])
                frame = self._load_single_frame(video_path, timestep, for_training)
                
                if frame is not None:
                    self.frame_cache.set(cache_key, frame)
                    frames.append(frame)
            
            return torch.stack(frames) if frames else None
        else:
            # Handle single camera index
            cache_key = (int(cam_idx), int(timestep), for_training)
            
            # Check cache first
            cached_frame = self.frame_cache.get(cache_key)
            if cached_frame is not None:
                return cached_frame
            
            # Check if in preloaded video
            if self.gt_video is not None and timestep < self.gt_video.shape[0]:
                frame = self.gt_video[timestep, cam_idx]
                if for_training:
                    frame = frame.float() / 255.0
                return frame
            
            # Load from disk
            dynamic_dir = os.path.join(self.sequence_dir, "dynamic")
            if not os.path.exists(dynamic_dir):
                return None
            
            video_files = sorted(f for f in os.listdir(dynamic_dir) if f.endswith('.mp4'))
            if cam_idx >= len(video_files):
                return None
            
            video_path = os.path.join(dynamic_dir, video_files[cam_idx])
            frame = self._load_single_frame(video_path, timestep, for_training)
            
            if frame is not None:
                self.frame_cache.set(cache_key, frame)
            
            return frame

class GMDataset(Dataset):
    """Dataset for Gaussian Model sequences."""
    def __init__(self, config: Config, num_sequences: int, dataset_path: str):
        self.config = config
        self.dataset_path = dataset_path
        
        # Find sequence directories in the specified path
        self.scene_dirs = sorted(
            os.path.join(dataset_path, d)
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith("sequence")
        )[:num_sequences]
        
        if not self.scene_dirs:
            raise ValueError(
                f"No scene directories found in {dataset_path}. "
                f"Make sure your data folder has subdirectories named like 'sequence0001', 'sequence0002', etc."
            )
            
        self.scenes_data = [SceneData(config, d) for d in self.scene_dirs]
        
    def __len__(self) -> int:
        return len(self.scenes_data)
        
    def __getitem__(self, idx: int) -> Dict:
        scene_data = self.scenes_data[idx]
        item = {
            "seq_idx": idx,
            "scene": scene_data.scene,
            "GM0": scene_data.gm0,
            "gt_image": scene_data.gt_image,
            "gt_video": scene_data.gt_video,
        }
        
        if scene_data.z0 is not None:
            item["z0"] = scene_data.z0
            
        return item

# --------------------------
# Training Utilities
# --------------------------
def custom_collate(batch: List[Dict]) -> Dict:
    """Custom collate function for handling variable-sized point clouds."""
    collated = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key in ['scene', 'GM0']:
            collated[key] = [sample[key] for sample in batch]
        elif key == 'z0':
            z0_list = [sample[key] for sample in batch]
            max_points = max(z.shape[0] for z in z0_list)
            feature_dim = z0_list[0].shape[-1]
            
            padded = torch.zeros(len(z0_list), max_points, feature_dim, 
                                dtype=z0_list[0].dtype, device=z0_list[0].device)
            mask = torch.zeros(len(z0_list), max_points, dtype=torch.bool, 
                                device=z0_list[0].device)
            
            for i, z in enumerate(z0_list):
                padded[i, :z.shape[0]] = z
                mask[i, :z.shape[0]] = True
                
            collated[key] = padded
            collated[f'{key}_mask'] = mask
        elif key == 'gt_video':
            collated[key] = [sample[key] for sample in batch]
        else:
            try:
                collated[key] = default_collate([sample[key] for sample in batch])
            except:
                collated[key] = [sample[key] for sample in batch]
                
    return collated

def extract_features(GM0: GaussianModel, GM1: GaussianModel) -> torch.Tensor:
    """Extract position, velocity, rotation and angular velocity features."""
    xyz0, rot0 = GM0.get_xyz, GM0.get_rotation
    xyz1, rot1 = GM1.get_xyz, GM1.get_rotation
    
    vel = xyz1 - xyz0
    q_diff = QuaternionOps.multiply(rot1, QuaternionOps.inverse(rot0))
    omega = 2 * q_diff[..., 1:] / (q_diff[..., 0:1] + 1e-6)
    
    return torch.cat([xyz0, vel, rot0, omega], dim=-1)


def visualize_results(loader: DataLoader, processor: nn.Module, config: Config,
                    device: torch.device, background: torch.Tensor,
                    segment_length: int, dataset: GMDataset, step: int,
                    is_test: bool = False) -> Dict[str, Any]:
    """Generate and log visualization videos."""
    video_logs = {}
    batch = next(iter(loader))
    
    with torch.no_grad():
        for i in range(len(batch['GM0'])):
            scene = batch['scene'][i]
            GM0 = batch['GM0'][i]
            
            # Get initial state
            xyz0 = GM0._xyz.detach()
            rot0 = GM0._rotation.detach()
            
            # Concatenate position and rotation
            z0 = torch.cat([xyz0, rot0], dim=-1).unsqueeze(0)  # Shape: [1, num_points, 7]
            
            # Generate trajectory
            z_traj = processor(z0, torch.linspace(
                0, segment_length/config.framerate, 
                segment_length, device=device
            ))[..., :13]
            
            for cam_idx in range(3):  # First 10 cameras
                # Load ground truth
                gt_frames = []
                for t in range(segment_length):
                    frame = dataset.scenes_data[i].get_frame(t, cam_idx, for_training=False)
                    if frame is not None:
                        gt_frames.append(frame.cpu())
                
                if not gt_frames:
                    continue
                    
                gt_video = torch.stack(gt_frames).numpy()  # [T, 3, H, W]
                
                # Generate predictions
                pred_frames = []
                for t in range(segment_length):
                    xyz = z_traj[t, 0, :, :3]
                    quat = F.normalize(z_traj[t, 0, :, 3:7], dim=-1)
                    
                    temp_gaussians = GM0.clone()
                    temp_gaussians.update_gaussians(xyz, quat)
                    
                    render = render_batch(
                        [scene.getTrainCameraObjects()[cam_idx]],
                        temp_gaussians, config.pipeline, background
                    )["render"].squeeze(0).cpu().numpy()
                    
                    pred_frames.append(render)
                
                # Combine and log videos
                pred_video = np.stack(pred_frames)  # [T, 3, H, W]
                combined = create_side_by_side_video(pred_video, gt_video)
                prefix = "test" if is_test else "train"
                
                # Log videos in wandb format
                video_logs[f"{prefix}_comparison_seq{i}_cam{cam_idx}"] = wandb.Video(
                    combined, fps=config.wandb_fps, format="mp4"
                )
    
    if config.use_wandb:
        wandb.log(video_logs, step=step)
    return video_logs

def create_side_by_side_video(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison video.
    
    Args:
        pred: Predicted frames of shape [T, C, H, W] in [0, 1] range
        gt: Ground truth frames of shape [T, C, H, W] in [0, 255] range
        
    Returns:
        Combined video of shape [T, C, H, W*2] for wandb
    """
    # Ensure same dimensions
    T, C, H, W = pred.shape
    gt = gt[:T]  # Trim to same length
    
    if (H, W) != (gt.shape[2], gt.shape[3]):
        resized_gt = np.zeros((gt.shape[0], C, H, W), dtype=gt.dtype)
        for t in range(gt.shape[0]):
            resized_gt[t] = cv2.resize(gt[t].transpose(1, 2, 0), (W, H)).transpose(2, 0, 1)
        gt = resized_gt
    
    # Convert predicted frames to uint8 (they're in [0, 1] range)
    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)
    
    # Ground truth is already in uint8 [0, 255] range
    
    # Concatenate side by side along width dimension
    combined = np.concatenate([pred, gt], axis=3)  # [T, C, H, W*2]
    
    return combined

def map_gradients_to_rgb(gradients: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Map gradient components to RGB colors with magnitude-based intensity, normalized by Gaussian scales.
    
    Args:
        gradients: Tensor of shape [N, 3] containing xyz gradients
        scales: Tensor of shape [N, 3] containing xyz scales of Gaussians
        
    Returns:
        Tensor of shape [N, 3] containing RGB colors in [0, 1] range
    """
    # Normalize gradients by scales
    scale_norm = torch.norm(scales, dim=1, keepdim=True)
    normalized_grads = gradients / (scale_norm + 1e-6)  # Add epsilon to avoid division by zero
    
    # Compute magnitude for intensity
    grad_norm = torch.norm(normalized_grads, dim=1, keepdim=True)
    grad_norm = torch.clamp(grad_norm, min=1e-6)  # Avoid division by zero
    
    # Normalize gradients to [-1, 1] range for direction
    normalized_grads = normalized_grads / grad_norm
    
    # Map to [0, 1] range for color
    rgb = (normalized_grads + 1) / 2
    
    # Scale by magnitude (normalized to [0, 1] range)
    max_norm = grad_norm.max()
    min_norm = grad_norm.min()
    norm_range = max_norm - min_norm
    if norm_range > 0:
        intensity = (grad_norm - min_norm) / norm_range
    else:
        intensity = torch.ones_like(grad_norm)
    
    # Apply intensity to colors
    rgb = rgb * intensity * 100
    return rgb

def log_loss_components(loss_components: Dict[str, List[float]], seq_idx: int, step: int, config: Config):
    """Log individual loss components to wandb.
    
    Args:
        loss_components: Dictionary containing lists of loss values for each component
        seq_idx: Index of the sequence
        step: Current training step
        config: Configuration object
    """
    if config.use_wandb:
        wandb.log({
            f"gradient_mse_loss_seq{seq_idx}": np.mean(loss_components['mse']),
            f"gradient_ssim_loss_seq{seq_idx}": np.mean(loss_components['ssim']),
            f"gradient_lpips_loss_seq{seq_idx}": np.mean(loss_components['lpips'])
        }, step=step)

def smooth_gradients(gradients: torch.Tensor, positions: torch.Tensor, radius: float = 0.1, num_neighbors: int = 5) -> torch.Tensor:
    """Smooth gradients based on spatial proximity of gaussians.
    
    Args:
        gradients: Tensor of shape [N, 3] containing xyz gradients
        positions: Tensor of shape [N, 3] containing xyz positions
        radius: Maximum distance to consider for neighbors
        num_neighbors: Maximum number of neighbors to consider
        
    Returns:
        Smoothed gradients of shape [N, 3]
    """
    N = positions.shape[0]
    
    # Compute pairwise distances between all gaussians
    # Reshape positions for broadcasting: [N, 1, 3] - [1, N, 3] = [N, N, 3]
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
    distances = torch.norm(pos_diff, dim=2)  # [N, N]
    
    # Create mask for neighbors within radius
    neighbor_mask = distances < radius  # [N, N]
    
    # For each gaussian, get indices of its neighbors
    neighbor_indices = torch.where(neighbor_mask)  # Tuple of (row_indices, col_indices)
    
    # Initialize smoothed gradients
    smoothed_grads = torch.zeros_like(gradients)
    
    # For each gaussian, average its gradient with its neighbors
    for i in range(N):
        # Get indices of neighbors for this gaussian
        neighbor_cols = neighbor_indices[1][neighbor_indices[0] == i]
        
        # Limit number of neighbors if needed
        if len(neighbor_cols) > num_neighbors:
            # Sort by distance and take closest neighbors
            neighbor_dists = distances[i, neighbor_cols]
            _, closest_indices = torch.topk(neighbor_dists, num_neighbors, largest=False)
            neighbor_cols = neighbor_cols[closest_indices]
        
        # Include the gaussian itself in the averaging
        indices_to_average = torch.cat([torch.tensor([i], device=gradients.device), neighbor_cols])
        
        # Average gradients
        smoothed_grads[i] = gradients[indices_to_average].mean(dim=0)
    
    return smoothed_grads

def visualize_gradients(loader: DataLoader, processor: nn.Module, config: Config,
                    device: torch.device, background: torch.Tensor,
                    segment_length: int, dataset: GMDataset, step: int, lpips_model: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Generate and log gradient visualization videos."""
    video_logs = {}
    batch = next(iter(loader))
    
    for i in range(len(batch['GM0'])):
        scene = batch['scene'][i]
        GM0 = batch['GM0'][i]
        
        # Get initial state
        xyz0 = GM0._xyz.detach()
        rot0 = GM0._rotation.detach()
        
        # Concatenate position and rotation
        z0 = torch.cat([xyz0, rot0], dim=-1).unsqueeze(0)  # Shape: [1, num_points, 7]
        
        # Generate trajectory
        z_traj = processor(z0, torch.linspace(
            0, segment_length/config.framerate, 
            segment_length, device=device
        ))[..., :13]
        
        # Generate visualization frames with gradient colors
        grad_frames = []
        
        for t in range(segment_length):
            # Create fresh gaussian model for this timestep
            temp_gaussians = GM0.clone()
            
            # Update gaussians while maintaining gradient connection
            xyz = z_traj[t, 0, :, :3]
            quat = F.normalize(z_traj[t, 0, :, 3:7], dim=-1)
            
            # Set requires_grad before updating positions
            temp_gaussians._xyz.requires_grad_(True)
            
            # Update positions while maintaining gradient connection
            temp_gaussians._xyz.data = xyz
            temp_gaussians._rotation.data = quat
            
            # Compute loss for current timestep
            losses = []
            for cam_idx in range(10):  # First 10 cameras
                gt = dataset.scenes_data[i].get_frame(t, cam_idx, for_training=True)
                if gt is not None:
                    gt = gt.to(device)
                    render = render_batch(
                        [scene.getTrainCameraObjects()[cam_idx]],
                        temp_gaussians, config.pipeline, background
                    )["render"]
                    
                    # Compute total loss
                    loss, _ = compute_loss(render, gt, lpips_model)
                    losses.append(loss)
            
            if losses:
                total_loss = torch.mean(torch.stack(losses))
                total_loss.backward()
                pos_grads = temp_gaussians._xyz.grad
                if pos_grads is not None:
                    # Get Gaussian scales
                    scales = temp_gaussians.get_scaling.detach()
                    
                    # Smooth the gradients based on spatial proximity
                    # smoothed_grads = smooth_gradients(
                    #     pos_grads,
                    #     xyz,
                    #     radius=0.1,  # Adjust this value based on your scene scale
                    #     num_neighbors=5  # Adjust this value based on your needs
                    # )
                    # temp_gaussians._xyz.grad = smoothed_grads
                    
                    # Render with gradient colors normalized by scales
                    grad_colors = map_gradients_to_rgb(pos_grads, scales)
                    grad_render = render_batch(
                        [scene.getTrainCameraObjects()[0]],
                        temp_gaussians, config.pipeline, background,
                        override_color=grad_colors  # Colors in [0, 1]
                    )["render"].squeeze(0).detach().cpu().numpy()
                    
                    grad_frames.append(grad_render)
        
        if grad_frames:
            # Stack frames into video
            grad_video = np.stack(grad_frames)  # [T, 3, H, W]
            
            # Convert to uint8 in [0, 255] range for video
            grad_video = (grad_video * 255).astype(np.uint8)
            
            # Log video in wandb format
            video_logs[f"gradient_visualization_seq{i}"] = wandb.Video(
                grad_video, fps=config.wandb_fps, format="mp4"
            )
    
    if config.use_wandb:
        wandb.log(video_logs, step=step)
    return video_logs

def map_velocities_to_rgb(velocities: torch.Tensor) -> torch.Tensor:
    """Map velocity components to RGB colors with magnitude-based intensity.
    
    Args:
        velocities: Tensor of shape [N, 3] containing xyz velocities
        
    Returns:
        Tensor of shape [N, 3] containing RGB colors in [0, 1] range
    """
    # Compute magnitude for intensity
    vel_norm = torch.norm(velocities, dim=1, keepdim=True)
    vel_norm = torch.clamp(vel_norm, min=1e-6)  # Avoid division by zero
    
    # Normalize velocities to [-1, 1] range for direction
    normalized_vels = velocities / vel_norm
    
    # Map to [0, 1] range for color
    rgb = (normalized_vels + 1) / 2
    
    # Scale by magnitude (normalized to [0, 1] range)
    max_norm = vel_norm.max()
    min_norm = vel_norm.min()
    norm_range = max_norm - min_norm
    if norm_range > 0:
        intensity = (vel_norm - min_norm) / norm_range
    else:
        intensity = torch.ones_like(vel_norm)
    
    # Apply intensity to colors
    rgb = rgb * intensity
    return rgb

def visualize_velocities(loader: DataLoader, processor: nn.Module, config: Config,
                    device: torch.device, background: torch.Tensor,
                    segment_length: int, dataset: GMDataset, step: int) -> Dict[str, Any]:
    """Generate and log velocity visualization videos."""
    video_logs = {}
    batch = next(iter(loader))
    
    for i in range(len(batch['GM0'])):
        scene = batch['scene'][i]
        GM0 = batch['GM0'][i]
        
        # Get initial state
        xyz0 = GM0._xyz.detach()
        rot0 = GM0._rotation.detach()
        
        # Concatenate position and rotation
        z0 = torch.cat([xyz0, rot0], dim=-1).unsqueeze(0)  # Shape: [1, num_points, 7]
        
        # Generate trajectory
        z_traj = processor(z0, torch.linspace(
            0, segment_length/config.framerate, 
            segment_length, device=device
        ))[..., :13]
        
        # Generate visualization frames with velocity colors
        vel_frames = []
        for t in range(segment_length):
            # Create fresh gaussian model for this timestep
            temp_gaussians = GM0.clone()
            
            # Update gaussians
            xyz = z_traj[t, 0, :, :3]
            quat = F.normalize(z_traj[t, 0, :, 3:7], dim=-1)
            vel = z_traj[t, 0, :, 7:10]  # Get velocities
            
            # Update positions
            temp_gaussians._xyz.data = xyz
            temp_gaussians._rotation.data = quat
            
            # Map velocities to colors
            vel_colors = map_velocities_to_rgb(vel)
            
            # Render with velocity colors for first camera only
            vel_render = render_batch(
                [scene.getTrainCameraObjects()[0]],
                temp_gaussians, config.pipeline, background,
                override_color=vel_colors
            )["render"].squeeze(0).detach().cpu().numpy()
            
            vel_frames.append(vel_render)
        
        if vel_frames:
            # Stack frames into video
            vel_video = np.stack(vel_frames)  # [T, 3, H, W]
            
            # Scale colors from [0,1] to [0,255] range
            vel_video = (vel_video * 255).astype(np.uint8)
            
            # Log video in wandb format
            video_logs[f"velocity_visualization_seq{i}"] = wandb.Video(
                vel_video, fps=config.wandb_fps, format="mp4"
            )
    
    if config.use_wandb:
        wandb.log(video_logs, step=step)
    return video_logs

def compute_ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute SSIM loss between predicted and ground truth images.
    
    Args:
        pred: Predicted image tensor of shape [B, C, H, W] or [C, H, W]
        gt: Ground truth image tensor of shape [B, C, H, W] or [C, H, W]
        
    Returns:
        SSIM loss tensor (1 - SSIM)
    """
    # Ensure input is in [0, 1] range
    pred = torch.clamp(pred, 0, 1)
    gt = torch.clamp(gt, 0, 1)
    
    # Compute SSIM
    ssim = F.cosine_similarity(pred, gt, dim=1)
    return 1 - ssim.mean()

def compute_lpips_loss(pred: torch.Tensor, gt: torch.Tensor, lpips_model: nn.Module) -> torch.Tensor:
    """Compute LPIPS loss between predicted and ground truth images.
    
    Args:
        pred: Predicted image tensor of shape [B, C, H, W] or [C, H, W]
        gt: Ground truth image tensor of shape [B, C, H, W] or [C, H, W]
        lpips_model: Pre-trained LPIPS model
        
    Returns:
        LPIPS loss tensor
    """
    # Ensure input is in [-1, 1] range for LPIPS
    pred = 2 * pred - 1
    gt = 2 * gt - 1
    
    # Compute LPIPS loss
    return lpips_model(pred, gt).mean()

def compute_loss(pred: torch.Tensor, gt: torch.Tensor, lpips_model: Optional[nn.Module] = None,
                reduction: str = 'mean', weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute combined loss between predicted and ground truth images.
    
    Args:
        pred: Predicted image tensor of shape [B, C, H, W] or [C, H, W]
        gt: Ground truth image tensor of shape [B, C, H, W] or [C, H, W]
        lpips_model: Optional pre-trained LPIPS model
        reduction: Reduction method for the loss ('mean', 'none', or 'sum')
        weights: Optional dictionary of loss weights
        
    Returns:
        Tuple containing:
        - Combined loss tensor
        - Dictionary of individual loss components (mse, ssim, lpips)
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'mse': 1.0,      # MSE is already small (0.0002)
            'ssim': 0.001,   # Scale down SSIM (0.75) to match MSE scale
            'lpips': 0.04    # Scale down LPIPS (0.0052) to match MSE scale
        }
    
    # MSE loss
    mse_loss = F.mse_loss(pred, gt, reduction=reduction)
    
    # SSIM loss
    ssim_loss = torch.tensor(0.0, device=pred.device)
    # ssim_loss = compute_ssim_loss(pred, gt)
    
    # LPIPS loss if model is provided
    lpips_loss = torch.tensor(0.0, device=pred.device)
    # if lpips_model is not None:
    #     lpips_loss = compute_lpips_loss(pred, gt, lpips_model)
    
    # Store individual losses
    loss_components = {
        'mse': weights['mse'] * mse_loss,
        'ssim': weights['ssim'] * ssim_loss,
        'lpips': weights['lpips'] * lpips_loss
    }
    
    # Combine losses with weights
    total_loss = (
        weights['mse'] * mse_loss +
        weights['ssim'] * ssim_loss +
        weights['lpips'] * lpips_loss
    )

    return total_loss, loss_components

def compute_per_pixel_loss(pred: torch.Tensor, gt: torch.Tensor, lpips_model: Optional[nn.Module] = None) -> torch.Tensor:
    """Compute per-pixel combined loss between predicted and ground truth images.
    
    Args:
        pred: Predicted image tensor of shape [B, C, H, W] or [C, H, W]
        gt: Ground truth image tensor of shape [B, C, H, W] or [C, H, W]
        lpips_model: Optional pre-trained LPIPS model
        
    Returns:
        Per-pixel loss tensor of shape [H, W] (averaged across channels)
    """
    # For per-pixel loss, we only use MSE as other losses are global
    per_pixel_loss = F.mse_loss(pred, gt, reduction='none')
    return per_pixel_loss.mean(dim=0)  # Average across channels

def visualize_loss(loader: DataLoader, processor: nn.Module, config: Config,
                    device: torch.device, background: torch.Tensor,
                    segment_length: int, dataset: GMDataset, step: int) -> Dict[str, Any]:
    """Generate and log per-pixel loss visualization videos for the first camera."""
    video_logs = {}
    batch = next(iter(loader))
    
    for i in range(len(batch['GM0'])):
        scene = batch['scene'][i]
        GM0 = batch['GM0'][i]
        
        # Get initial state
        xyz0 = GM0._xyz.detach()
        rot0 = GM0._rotation.detach()
        
        # Concatenate position and rotation
        z0 = torch.cat([xyz0, rot0], dim=-1).unsqueeze(0)  # Shape: [1, num_points, 7]
        
        # Generate trajectory
        z_traj = processor(z0, torch.linspace(
            0, segment_length/config.framerate, 
            segment_length, device=device
        ))[..., :13]
        
        # Get ground truth frames for first camera
        gt_frames = []
        for t in range(segment_length):
            gt = dataset.scenes_data[i].get_frame(t, 0, for_training=True)
            if gt is not None:
                gt_frames.append(gt)
        
        if gt_frames:
            gt_video = torch.stack(gt_frames).to(device)  # [T, 3, H, W]
            
            # Generate visualization frames with loss values
            loss_frames = []
            for t in range(segment_length):
                # Create fresh gaussian model for this timestep
                temp_gaussians = GM0.clone()
                
                # Update gaussians
                xyz = z_traj[t, 0, :, :3]
                quat = F.normalize(z_traj[t, 0, :, 3:7], dim=-1)
                
                # Update positions
                temp_gaussians._xyz.data = xyz
                temp_gaussians._rotation.data = quat
                
                # Render for first camera
                render = render_batch(
                    [scene.getTrainCameraObjects()[0]],
                    temp_gaussians, config.pipeline, background
                )["render"]
                
                # Compute per-pixel loss
                grayscale_loss = compute_per_pixel_loss(render, gt_video[t])
                
                # Normalize loss to [0, 1] range
                max_loss = grayscale_loss.max()
                min_loss = grayscale_loss.min()
                loss_range = max_loss - min_loss
                if loss_range > 0:
                    normalized_loss = (grayscale_loss - min_loss) / loss_range
                else:
                    normalized_loss = torch.zeros_like(grayscale_loss)
                
                # Convert to numpy and create RGB frame
                loss_frame = normalized_loss.detach().cpu().numpy()
                
                # Convert to uint8 in [0, 255] range
                loss_frame = (loss_frame * 255).astype(np.uint8)
                
                loss_frames.append(loss_frame)
            
            if loss_frames:
                # Stack frames into video
                loss_video = np.stack(loss_frames)  # [T, H, W, 3]
                
                # Log video in wandb format
                video_logs[f"loss_visualization_seq{i}"] = wandb.Video(
                    loss_video, fps=config.wandb_fps, format="mp4"
                )
    
    if config.use_wandb:
        wandb.log(video_logs, step=step)
    return video_logs

def visualize_downscaled_gaussians(loader: DataLoader, processor: nn.Module, config: Config,
                    device: torch.device, background: torch.Tensor,
                    segment_length: int, dataset: GMDataset, step: int,
                    is_test: bool = False) -> Dict[str, Any]:
    """Generate and log visualization videos with downscaled gaussians."""
    video_logs = {}
    batch = next(iter(loader))
    
    with torch.no_grad():
        for i in range(len(batch['GM0'])):
            scene = batch['scene'][i]
            GM0 = batch['GM0'][i]
            
            # Get initial state
            xyz0 = GM0._xyz.detach()
            rot0 = GM0._rotation.detach()
            
            # Concatenate position and rotation
            z0 = torch.cat([xyz0, rot0], dim=-1).unsqueeze(0)  # Shape: [1, num_points, 7]
            
            # Generate trajectory
            z_traj = processor(z0, torch.linspace(
                0, segment_length/config.framerate, 
                segment_length, device=device
            ))[..., :13]
            
            for cam_idx in range(3):  # First 10 cameras
                # Load ground truth
                gt_frames = []
                for t in range(segment_length):
                    frame = dataset.scenes_data[i].get_frame(t, cam_idx, for_training=False)
                    if frame is not None:
                        gt_frames.append(frame.cpu())
                
                if not gt_frames:
                    continue
                    
                gt_video = torch.stack(gt_frames).numpy()  # [T, 3, H, W]
                
                # Generate predictions with downscaled gaussians
                pred_frames = []
                for t in range(segment_length):
                    xyz = z_traj[t, 0, :, :3]
                    quat = F.normalize(z_traj[t, 0, :, 3:7], dim=-1)
                    
                    temp_gaussians = GM0.clone()
                    # Update gaussians with downscaled scaling
                    temp_gaussians.update_gaussians(xyz, quat, scaling_factor=4.0)
                    
                    render = render_batch(
                        [scene.getTrainCameraObjects()[cam_idx]],
                        temp_gaussians, config.pipeline, background
                    )["render"].squeeze(0).cpu().numpy()
                    
                    pred_frames.append(render)
                
                # Combine and log videos
                pred_video = np.stack(pred_frames)  # [T, 3, H, W]
                combined = create_side_by_side_video(pred_video, gt_video)
                prefix = "test" if is_test else "train"
                
                # Log videos in wandb format
                video_logs[f"{prefix}_downscaled_comparison_seq{i}_cam{cam_idx}"] = wandb.Video(
                    combined, fps=config.wandb_fps, format="mp4"
                )
    
    if config.use_wandb:
        wandb.log(video_logs, step=step)
    return video_logs

def apply_annealing_noise(z0: torch.Tensor, iteration: int, config: Config) -> torch.Tensor:
    """Apply annealing noise to the input state z0.
    
    Args:
        z0: Input state tensor of shape [batch_size, num_points, feature_dim]
        iteration: Current training iteration
        config: Configuration object containing noise parameters
        
    Returns:
        Noised state tensor of same shape as input
    """
    # Compute noise scale that decreases linearly until noise_end_iter
    noise_factor = 1.0 - min(1.0, iteration / config.noise_end_iter)
    
    # Generate random noise for position and rotation components
    pos_noise = torch.randn_like(z0[..., :3], device=z0.device) * config.noise_scale * noise_factor
    rot_noise = torch.randn_like(z0[..., 3:7], device=z0.device) * config.noise_scale * noise_factor * 0.1  # Less noise for rotations
    
    # Create noised version
    noised_z0 = z0.clone()
    noised_z0[..., :3] = z0[..., :3] + pos_noise
    noised_z0[..., 3:7] = F.normalize(z0[..., 3:7] + rot_noise, dim=-1)  # Ensure quaternions stay normalized
    
    return noised_z0

# --------------------------
# Training Loop
# --------------------------
def train(config: Config):
    """Main training procedure."""
    device = torch.device(config.device)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    t_span = torch.linspace(0, config.num_timestep_samples/config.framerate, 
                            config.num_timestep_samples, device=device)
    
    # Initialize datasets with correct paths
    train_dataset = GMDataset(config, config.num_train_sequences, f"{config.dataset_path}/train")
    test_dataset = GMDataset(config, config.num_test_sequences, f"{config.dataset_path}/test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=custom_collate)
    
    # Initialize model and optimizer
    processor = LatentNeuralODE(
        num_points=train_dataset.scenes_data[0].gm0._xyz.shape[0],
        latent_dim=64,
        atol=config.ode_atol,
        rtol=config.ode_rtol,
        method=config.ode_method,
        config=config
    ).to(device)
    
    # Initialize LPIPS model if available
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
    except ImportError:
        print("LPIPS not available. Will use only MSE and SSIM losses.")
        lpips_model = None
    
    # Create parameter groups for different learning rates
    decaying_parameters = [processor.initial_velocity, processor.initial_omega]#, processor.initial_position_offset, processor.initial_rotation_offset]
    decaying_param_ids = [id(p) for p in decaying_parameters]
    other_params = [p for p in processor.parameters() if id(p) not in decaying_param_ids]
    
    # Configure optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': other_params, 'lr': config.learning_rate},
        {'params': decaying_parameters, 'lr': config.learning_rate * config.decaying_params_lr_scale}
    ])
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(project="latent_node", config=config.__dict__)
    
    # Create progress bar for epochs with loss display
    pbar = tqdm(range(config.epochs), desc="Training", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - Loss: {postfix}')
    
    # Current decaying parameters learning rate scale factor
    decaying_lr_scale = config.decaying_params_lr_scale
    
    # Training loop
    for epoch in pbar:
        torch.cuda.empty_cache()
        processor.train()
        epoch_loss = []
        reg_losses = []
        
        # Initialize gradient tracking lists for this epoch
        decaying_grad_norms = []
        other_grad_norms = []
        
        # Initialize loss components tracking for this epoch
        epoch_loss_components = {
            'mse': [],
            'ssim': [],
            'lpips': []
        }
        
        # Update learning rate for decaying parameters
        decaying_lr_scale *= config.decaying_params_lr_decay
        optimizer.param_groups[1]['lr'] = config.learning_rate * decaying_lr_scale
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            GM0 = batch['GM0'][0]  # Use first sequence's static model
            xyz0 = GM0._xyz.detach()
            rot0 = GM0._rotation.detach()
            z0 = torch.cat([xyz0, rot0], dim=-1).unsqueeze(0)
            
            
            processor.func.nfe = 0
            z_traj = processor(z0, t_span, epoch) # num_timesteps x num_points x 13+latent_dim
            
            # Compute photometric loss
            batch_loss = []
            for be in range(len(batch['GM0'])):
                scene = batch['scene'][be]
                seq_idx = batch['seq_idx'][be].item()
                cams = scene.getTrainCameraObjects()[:10]
                
                # Temporary gaussians for rendering
                temp_gaussians = GM0.clone()
                
                # Compute loss for each timestep
                timestep_losses = []
                for t in range(len(t_span)):
                    # Update gaussians and render
                    xyz = z_traj[t, be, :, :3]
                    quat = F.normalize(z_traj[t, be, :, 3:7], dim=-1)
                    temp_gaussians.update_gaussians(xyz, quat)
                    
                    render = render_batch(cams, temp_gaussians, config.pipeline, background)["render"]
                    gt = train_dataset.scenes_data[seq_idx].get_frame(t, slice(0, 10)).to(device)
                    
                    if gt is not None:
                        # Compute total loss and get components
                        loss, loss_components = compute_loss(render, gt, lpips_model)
                        
                        # Store loss components
                        epoch_loss_components['mse'].append(loss_components['mse'].item())
                        epoch_loss_components['ssim'].append(loss_components['ssim'].item())
                        epoch_loss_components['lpips'].append(loss_components['lpips'].item())
                        
                        timestep_losses.append(loss)
                
                if timestep_losses:
                    batch_loss.append(torch.mean(torch.stack(timestep_losses)))
            
            if batch_loss:
                # Backward pass with regularization
                loss = torch.mean(torch.stack(batch_loss))
                
                # Add regularization term if present
                if hasattr(processor.func, 'reg_loss') and processor.func.reg_loss != 0:
                    reg_loss = processor.func.reg_loss
                    total_loss = loss + config.reg_weight * reg_loss
                    reg_losses.append(reg_loss.item())
                else:
                    total_loss = loss
                
                total_loss.backward()
                
                # Smooth position gradients for each timestep
                # for t in range(len(t_span)):
                #     xyz = z_traj[t, 0, :, :3].detach()  # Get positions for this timestep
                #     if temp_gaussians._xyz.grad is not None:
                #         # Smooth the gradients based on spatial proximity
                #         smoothed_grads = smooth_gradients(
                #             temp_gaussians._xyz.grad,
                #             xyz,
                #             radius=0.1,  # Adjust this value based on your scene scale
                #             num_neighbors=5  # Adjust this value based on your needs
                #         )
                #         temp_gaussians._xyz.grad = smoothed_grads
                
                # Calculate gradient norms for monitoring
                decaying_grad_norm = 0.0
                for param in decaying_parameters:
                    if param.grad is not None:
                        decaying_grad_norm += torch.norm(param.grad).item() ** 2
                decaying_grad_norm = decaying_grad_norm ** 0.5
                decaying_grad_norms.append(decaying_grad_norm)
                
                other_grad_norm = 0.0
                for param in other_params:
                    if param.grad is not None:
                        other_grad_norm += torch.norm(param.grad).item() ** 2
                other_grad_norm = other_grad_norm ** 0.5
                other_grad_norms.append(other_grad_norm)
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(processor.parameters(), config.grad_clip_value)
                
                optimizer.step()
                epoch_loss.append(loss.item())
        
        # Logging
        if epoch_loss:
            avg_loss = np.mean(epoch_loss)
            avg_reg_loss = np.mean(reg_losses) if reg_losses else 0.0
            avg_decaying_grad_norm = np.mean(decaying_grad_norms) if decaying_grad_norms else 0.0
            avg_other_grad_norm = np.mean(other_grad_norms) if other_grad_norms else 0.0
            
            if config.use_wandb:
                # Log epoch metrics
                wandb.log({
                    "epoch_loss": avg_loss,
                    "reg_loss": avg_reg_loss,
                    "nfe": processor.func.nfe,
                    "len(t_span)": len(t_span),
                    "initial_velocity_norm": torch.norm(processor.initial_velocity).item(),
                    "initial_omega_norm": torch.norm(processor.initial_omega).item(),
                    "decaying_lr_scale": decaying_lr_scale,
                    "decaying_grad_norm": avg_decaying_grad_norm,
                    "other_grad_norm": avg_other_grad_norm,
                    # Log average loss components for the epoch
                    "mse_loss": np.mean(epoch_loss_components['mse']),
                    "ssim_loss": np.mean(epoch_loss_components['ssim']),
                    "lpips_loss": np.mean(epoch_loss_components['lpips'])
                }, step=epoch)
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f'{avg_loss:.6f}', 
                'nfe': f'{processor.func.nfe}'
            })
        
        # Visualization
        if epoch % config.train_visualization_interval == 0:
            visualize_results(train_loader, processor, config, device, 
                                background, len(t_span), train_dataset, epoch)
            visualize_gradients(train_loader, processor, config, device,
                                background, len(t_span), train_dataset, epoch, lpips_model)
            visualize_velocities(train_loader, processor, config, device,
                                background, len(t_span), train_dataset, epoch)
            visualize_loss(train_loader, processor, config, device,
                                background, len(t_span), train_dataset, epoch)
            visualize_downscaled_gaussians(train_loader, processor, config, device,
                                background, len(t_span), train_dataset, epoch)
        
        # Adjust time span (original approach)
        #if epoch % 100 == 0 and epoch > 0:
        if epoch > 0 and epoch % 200 == 0:
            config.num_timestep_samples += 1
            t_span = torch.linspace(0, config.num_timestep_samples/config.framerate, 
                                    config.num_timestep_samples, device=device)
    
    # Cleanup wandb if enabled
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    config = Config()
    train(config)