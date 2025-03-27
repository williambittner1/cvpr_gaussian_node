# gaussian_model_1.py:
# Runs NeuralODE on all gaussians individually.
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import copy
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from scipy.spatial import KDTree


def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

def get_cluster_colors(num_clusters):
    """
    Returns an array of distinct colors based on the number of clusters.
    The colors are chosen from the 'tab20' colormap and scaled to RGB values in [0, 255].
    """
    cmap = plt.get_cmap('tab20', num_clusters)  # Use a colormap with 20 distinct colors
    colors = [cmap(i)[:3] for i in range(num_clusters)]  # Extract RGB values
    colors = np.array(colors) * 255  # Scale to RGB [0, 255]
    return colors.astype(np.uint8)

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class GaussianModel:



    def __init__(self, sh_degree: int, device: str = 'cuda'):
        self.device = device  # Store the device

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        # Initialize tensors on the specified device
        self._xyz = nn.Parameter(torch.empty(0, 3, device=self.device, requires_grad=True))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3, device=self.device, requires_grad=False))
        self._features_rest = nn.Parameter(torch.empty(0, 15, 3, device=self.device, requires_grad=False))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=self.device, requires_grad=False))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=self.device, requires_grad=False))
        self._opacity = nn.Parameter(torch.empty(0, 1, device=self.device, requires_grad=False))
        self._semantic_feature = nn.Parameter(torch.empty(0, 1, 1, device=self.device, requires_grad=False))

        # Auxiliary attributes
        self.max_radii2D = torch.empty(0, device=self.device)
        self.xyz_gradient_accum = torch.empty(0, 1, device=self.device)
        self.denom = torch.empty(0, 1, device=self.device)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.initial_xyz = None


        # Clustering attributes (to be computed later).
        self.cluster_label = None                  # Tensor of cluster labels for each gaussian.
        self.cluster_control_points = None         # Tensor of shape [n_clusters, 3]
        self.cluster_control_orientations = None     # Tensor of shape [n_clusters, 4] (initially identity)
        self.relative_positions = None             # For each gaussian: offset from its cluster control point.



    def cluster_gaussians(self, n_clusters=3):
        """
        Cluster gaussians into 3 clusters based solely on their 3D positions.
        For each cluster the control point is defined as the mean of the positions
        of all gaussians in that cluster. The control orientation is initially the identity quaternion.
        Also, for each gaussian, compute and store its relative offset with respect to its cluster control point.
        """
        positions = self.get_xyz.detach().cpu().numpy()  # (N, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(positions)
        labels = kmeans.labels_  # (N,)
        self.cluster_label = torch.tensor(labels, device=self.device, dtype=torch.long)

        # Compute cluster control points (mean of positions in each cluster)
        cluster_control_points = []
        for cl in range(n_clusters):
            indices = np.where(labels == cl)[0]
            cluster_mean = np.mean(positions[indices], axis=0)
            cluster_control_points.append(cluster_mean)
        self.cluster_control_points = torch.tensor(cluster_control_points, device=self.device, dtype=self.get_xyz.dtype)

        # Set control orientations to identity for each cluster.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.get_xyz.dtype)
        self.cluster_control_orientations = identity_quat.repeat(n_clusters, 1)

        # Compute relative offsets: for each gaussian, relative offset = gaussian position - (cluster control point)
        N = positions.shape[0]
        rel_positions = []
        for i in range(N):
            cl = labels[i]
            rel = positions[i] - self.cluster_control_points[cl].cpu().numpy()
            rel_positions.append(rel)
        self.relative_positions = torch.tensor(rel_positions, device=self.device, dtype=self.get_xyz.dtype)




    def setup_functions(self):
        
        self.scaling_activation             = torch.exp
        self.scaling_inverse_activation     = torch.log
        self.covariance_activation          = build_covariance_from_scaling_rotation
        self.opacity_activation             = torch.sigmoid
        self.inverse_opacity_activation     = inverse_sigmoid
        self.rotation_activation            = torch.nn.functional.normalize

    def update_positions(self, positions):
        """
        Update the positions of the Gaussians.

        Args:
            positions (torch.Tensor): Tensor of shape (num_gaussians, 3)
        """

        new_xyz = positions.clone()
        
        self._xyz = new_xyz


    @property
    def colors(self):
        """
        Returns:
            torch.Tensor: Tensor of shape (N, 3) containing RGB colors for each Gaussian.
        """
        if not hasattr(self, '_colors'):
            # If colors have not been computed yet, initialize them to white
            self._colors = torch.ones((self._xyz.shape[0], 3), dtype=torch.uint8, device=self._xyz.device) * 255
        return self._colors
    


    def set_initial_positions(self):
        """
        Store the initial positions of the Gaussians.
        """
        self.initial_xyz = self._xyz.clone()

    def move_along_trajectory(self, translation_vector):
        """
        Update the positions of the Gaussians based on the translation vector.

        Args:
            translation_vector (torch.Tensor): Tensor of shape (3,) or (N, 3) representing the translation vector.
        """
        if translation_vector.dim() == 1:
            translation_vector = translation_vector.unsqueeze(0).expand_as(self._xyz)
        self._xyz = self.initial_xyz + translation_vector


    def update_gaussians_from_predictions(self, pred):
        """
        Update the positions and rotations of the Gaussians based on the predictions.

        Args:
            pred (torch.Tensor): Tensor of shape (num_gaussians, 7), where:
                - pred[:, :3]: positions (x, y, z)
                - pred[:, 3:7]: quaternion rotations (w, x, y, z)
        """
        new_xyz = pred[:, :3].clone()
        new_rotation = pred[:, 3:7].clone()


        if torch.norm(new_rotation, dim=-1).min().item() < 1e-8:
            print("Warning: Zero-length quaternion detected")
            
        # Normalize the quaternion rotations
        new_rotation = F.normalize(new_rotation, p=2, dim=-1)

        # Update positions and rotations
        self._xyz = new_xyz
        self._rotation = new_rotation


    def update_colors(self):
        """
        Update the colors of each Gaussian based on its cluster label.
        All Gaussians in the same cluster will have the same color.
        """
        if self.cluster_label is None:
            # Assign a default color (e.g., white) if no clustering has been done
            self._colors = torch.ones((self._xyz.shape[0], 3), dtype=torch.uint8, device=self._xyz.device) * 255
            return

        unique_labels = np.unique(self.cluster_label)
        num_clusters = len(unique_labels)

        # Generate distinct colors for each cluster
        color_map = get_cluster_colors(num_clusters)  # Shape: (num_clusters, 3)

        # Ensure cluster labels are from 0 to num_clusters-1
        # This is typically true for KMeans clustering
        if not np.array_equal(unique_labels, np.arange(num_clusters)):
            # Create a mapping from original labels to indices
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            indices = np.array([label_to_index[label] for label in self.cluster_label])
        else:
            indices = self.cluster_label

        # Map each Gaussian to its corresponding color
        colors = color_map[indices]  # Shape: (N, 3)

        # Convert to a Torch tensor
        self._colors = torch.tensor(colors, dtype=torch.uint8, device=self._xyz.device)




            

    
    def remove_all_gaussians(self):
        """
        Remove all Gaussians from the model by resetting all relevant tensors.
        """
        device = self._xyz.device if self._xyz.numel() > 0 else 'cuda'

        # Reset all tensors to empty
        self._xyz = nn.Parameter(torch.empty(0, 3, device=device, requires_grad=True))
        self._features_dc = nn.Parameter(torch.empty(0, 3, 1, device=device, requires_grad=True))
        self._features_rest = nn.Parameter(torch.empty(0, 3, (self.max_sh_degree + 1) ** 2 - 1, device=device, requires_grad=True))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device, requires_grad=True))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device, requires_grad=True))
        self._opacity = nn.Parameter(torch.empty(0, 1, device=device, requires_grad=True))
        self._semantic_feature = nn.Parameter(torch.empty(0, 1, 1, device=device, requires_grad=True))

        # Reset auxiliary attributes
        self.max_radii2D = torch.empty(0, device=device)
        self.xyz_gradient_accum = torch.empty(0, 1, device=device)
        self.denom = torch.empty(0, 1, device=device)

        # Reset optimizer
        if self.optimizer is not None:
            self.optimizer.param_groups = []
            self.optimizer.state = {}



    def add_gaussian(self, position, scale, rotation, opacity, color):
        """
        Manually add a Gaussian to the model with specified parameters.

        Args:
            position (torch.Tensor): Tensor of shape (3,) representing the position (x, y, z).
            scale (torch.Tensor): Tensor of shape (3,) representing the scaling factors.
            rotation (torch.Tensor): Tensor of shape (4,) representing the quaternion rotation (w, x, y, z).
            opacity (float): Opacity value (between 0 and 1).
            color (torch.Tensor): Tensor of shape (3,) representing the RGB color values (between 0 and 1).
        """
        # Ensure inputs are tensors on the correct device
        position = position.to(self.device).unsqueeze(0)  # Shape: (1, 3)
        scale = scale.to(self.device).unsqueeze(0)        # Shape: (1, 3)
        rotation = rotation.to(self.device).unsqueeze(0)  # Shape: (1, 4)
        color = color.to(self.device).unsqueeze(0)        # Shape: (1, 3)
        opacity = torch.tensor([opacity], device=self.device).unsqueeze(0)  # Shape: (1, 1)

        # Normalize rotation quaternion
        rotation = F.normalize(rotation, p=2, dim=-1)

        # Prepare features
        features_dc = color.unsqueeze(1)  # Shape: (1, 1, 3)
        features_rest = torch.zeros((1, 15, 3), device=self.device)  # Shape: (1, 15, 3)

        # Prepare semantic feature (set to zeros)
        semantic_feature = torch.zeros((1, 1, 1), device=self.device)  # Shape: (1, 1, 1)

        # Inverse activations for scaling and opacity
        scaling_inv = self.scaling_inverse_activation(scale)
        opacity_inv = self.inverse_opacity_activation(opacity)

        # Concatenate new Gaussian parameters to existing tensors
        self._xyz = nn.Parameter(torch.cat([self._xyz, position], dim=0))  # (N+1, 3)
        self._scaling = nn.Parameter(torch.cat([self._scaling, scaling_inv], dim=0))  # (N+1, 3)
        self._rotation = nn.Parameter(torch.cat([self._rotation, rotation], dim=0))  # (N+1, 4)
        self._opacity = nn.Parameter(torch.cat([self._opacity, opacity_inv], dim=0))  # (N+1, 1)
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, features_dc], dim=0))  # (N+1, 1, 3)
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, features_rest], dim=0))  # (N+1, 15, 3)
        self._semantic_feature = nn.Parameter(torch.cat([self._semantic_feature, semantic_feature], dim=0))  # (N+1,1,1)

        # Update auxiliary attributes
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(1, device=self.device)], dim=0)  # (N+1,)
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(1, 1, device=self.device)], dim=0)  # (N+1,1)
        self.denom = torch.cat([self.denom, torch.zeros(1, 1, device=self.device)], dim=0)  # (N+1,1)


 

    def select_key_gaussians(self):

        cluster_labels = self.cluster_label
        unique_clusters = np.unique(cluster_labels)
        key_gaussian_indices = []
        for cluster in unique_clusters:
            indices_in_cluster = np.where(cluster_labels == cluster)[0]
            key_gaussian_index = indices_in_cluster[0]  # Select the first Gaussian
            key_gaussian_indices.append(key_gaussian_index)
        return key_gaussian_indices
    


    def return_modified_gaussian_model(self, pred):
        """
        Create and return a new GaussianModel with updated positions and rotations based on pred,
        without modifying the current model.

        Args:
            pred (torch.Tensor): Tensor of shape (num_gaussians, 7), where the first 3 values are xyz positions,
                                and the next 4 values are quaternion rotations (w, x, y, z).

        Returns:
            GaussianModel: A new GaussianModel instance with updated Gaussians.
        """
        # Clone the current model to create a new instance
        new_model = self.clone()

        # Split pred into positions and rotations and clone them
        new_xyz = pred[:, :3].clone()  # Shape: (num_gaussians, 3)
        new_rotation = pred[:, 3:].clone()  # Shape: (num_gaussians, 4)

        # Normalize the quaternion rotations
        new_rotation = F.normalize(new_rotation, p=2, dim=-1)

        # Replace the parameters in new_model
        new_model._xyz = nn.Parameter(new_xyz)
        new_model._rotation = nn.Parameter(new_rotation)

        # No need to re-initialize the optimizer here unless you plan to optimize new_model

        # Return the new GaussianModel instance with updated Gaussians
        return new_model



    def compute_relative_transformations(self):
        """
        Compute and store the relative positions and rotations of child Gaussians
        with respect to their associated keypoint Gaussians.
        """
        N = self._xyz.shape[0]

        # Initialize tensors for relative positions and rotations
        self.relative_positions = torch.zeros(N, 3, device=self._xyz.device)
        self.relative_rotations = torch.zeros(N, 4, device=self._xyz.device)

        # Get positions and rotations
        p_all = self._xyz  # Shape: (N, 3)
        q_all = self._rotation  # Shape: (N, 4)

        # Get parent indices
        parent_indices = self.cluster_parent_gaussian_index  # Shape: (N,)

        # Compute relative positions and rotations
        p_parent = p_all[parent_indices]  # Shape: (N, 3)
        q_parent = q_all[parent_indices]  # Shape: (N, 4)

        # Compute p_rel and q_rel
        p_diff = p_all - p_parent  # Shape: (N, 3)
        p_rel = rotate_vector(quaternion_inverse(q_parent), p_diff)  # Shape: (N, 3)
        self.relative_positions = p_rel

        q_rel = quaternion_multiply(quaternion_inverse(q_parent), q_all)  # Shape: (N, 4)
        self.relative_rotations = F.normalize(q_rel, p=2, dim=-1)


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_feature,
            self.is_keygaussian,
            self.cluster_parent_gaussian_index,
            self.key_gaussian_indices
        )
    


    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom,
        opt_dict,
        self.spatial_lr_scale,
        self._semantic_feature,
        self.is_keygaussian,
        self.cluster_parent_gaussian_index,
        self.key_gaussian_indices) = model_args
        self.training_setup_0(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    @property
    def get_semantic_feature(self):
        return self._semantic_feature 
    
    def rewrite_semantic_feature(self, x):
        self._semantic_feature = x

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, semantic_feature_size : int, speedup: bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        if speedup: # speed up for Segmentation
            semantic_feature_size = int(semantic_feature_size/4)

        
        # Here we set _semantic_feature to zeros, as you mentioned you don't need it.
        self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], 1, 1).float().cuda()


        # self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], semantic_feature_size, 1).float().cuda() 
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(False))
        self._scaling = nn.Parameter(scales.requires_grad_(False))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._semantic_feature = nn.Parameter(self._semantic_feature.transpose(1, 2).contiguous().requires_grad_(False))
        


    def training_setup_0(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic_feature], 'lr':training_args.semantic_feature_lr, "name": "semantic_feature"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def training_setup_t(self, training_args):
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def get_initial_parameters(self):
        """
        Get the initial Gaussian parameters for t=0.

        Returns:
            torch.Tensor: A concatenated tensor of positions, rotations, and scalings
                          for the initial Gaussian parameters at t=0.
        """
        # Assuming you want to concatenate positions (_xyz), rotations (_rotation), and scaling (_scaling)
        # Modify this to include other parameters if needed
        initial_parameters = torch.cat([self._xyz, self._rotation], dim=1)
        return initial_parameters
    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        for i in range(self._semantic_feature.shape[1]*self._semantic_feature.shape[2]):  
            l.append('semantic_{}'.format(i))
        return l

    def _get_gaussian_parameters(self):
        # Concatenate the parameters you need into a single tensor
        # For example, concatenating position (_xyz), rotation (_rotation), and scaling (_scaling)
        params = torch.cat([self._xyz, self._rotation], dim=1)
        return params
    
    def save_ply_cluster(self, path):

        """
        Save the Gaussians as a PLY file with colors based on their cluster labels.

        Args:
        - path (str): The path to save the PLY file.
        """
        mkdir_p(os.path.dirname(path))

        # Extract XYZ and other attributes from the GaussianModel
        xyz = self._xyz.detach().cpu().numpy()  # Shape: (N, 3)
        normals = np.zeros_like(xyz)  # Set normals to zero for simplicity
        opacities = self._opacity.detach().cpu().numpy()  # Shape: (N, 1)
        scale = self._scaling.detach().cpu().numpy()  # Shape: (N, 3)
        rotation = self._rotation.detach().cpu().numpy()  # Shape: (N, 4)

        # Get cluster labels and map them to colors
        if self.cluster_label is None:
            cluster_labels = np.zeros((xyz.shape[0],), dtype=int)  # If no labels, set all to zero
        else:
            cluster_labels = self.cluster_label.astype(int)

        # Generate colors based on the cluster labels
        num_clusters = np.max(cluster_labels) + 1  # Number of unique clusters
        colors = get_cluster_colors(num_clusters)  # Get distinct colors for each cluster
        cluster_colors = colors[cluster_labels]  # Map cluster labels to RGB colors

        # Prepare the dtype for the PLY file
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # Adding color attributes here
        ]

        # Prepare the data to save
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # Assign values to the PLY elements
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        elements['red'] = cluster_colors[:, 0]
        elements['green'] = cluster_colors[:, 1]
        elements['blue'] = cluster_colors[:, 2]

        # Create a PlyElement and save to the PLY file
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"PLY file saved to {path}")

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        semantic_feature = self._semantic_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 

        # Get cluster labels and colors
        if self.cluster_label is None:
            cluster_labels = np.zeros((xyz.shape[0],), dtype=int)
        else:
            cluster_labels = self.cluster_label.cpu().numpy().astype(int)
        num_clusters = np.max(cluster_labels) + 1
        colors = get_cluster_colors(num_clusters)
        cluster_colors = colors[cluster_labels]  # Map labels to colors



        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
        semantic_feature = np.expand_dims(semantic_feature, axis=-1) 

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "semantic_feature": new_semantic_feature
        } 

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1, 1) 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic_feature) 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature) 

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def clone_wrong(self):
        """Return a deep copy of the gaussian model (used to create a dynamic copy)."""
        new_model = GaussianModel(self.max_sh_degree, device=self.device)
        new_model._xyz.data = self._xyz.data.clone()
        new_model._rotation.data = self._rotation.data.clone()
        if self.cluster_label is not None:
            new_model.cluster_label = self.cluster_label.clone()
            new_model.cluster_control_points = self.cluster_control_points.clone()
            new_model.cluster_control_orientations = self.cluster_control_orientations.clone()
            new_model.relative_positions = self.relative_positions.clone()
        return new_model
    

    def clone(self):
        new_gaussian = GaussianModel(self.max_sh_degree)
        new_gaussian.active_sh_degree = self.active_sh_degree
        new_gaussian.spatial_lr_scale = self.spatial_lr_scale


        # Clone tensors
        def clone_tensor(tensor):
            if isinstance(tensor, nn.Parameter):
                return nn.Parameter(tensor.clone().detach().requires_grad_(tensor.requires_grad))
            else:
                return tensor.clone().detach()
        
        new_gaussian.cluster_label = self.cluster_label.clone() if self.cluster_label is not None else None
        new_gaussian._xyz = clone_tensor(self._xyz)
        new_gaussian._features_dc = clone_tensor(self._features_dc)
        new_gaussian._features_rest = clone_tensor(self._features_rest)
        new_gaussian._scaling = clone_tensor(self._scaling)
        new_gaussian._rotation = clone_tensor(self._rotation)
        new_gaussian._opacity = clone_tensor(self._opacity)
        new_gaussian.max_radii2D = self.max_radii2D.clone().detach()
        new_gaussian.xyz_gradient_accum = self.xyz_gradient_accum.clone().detach()
        new_gaussian.denom = self.denom.clone().detach()
        new_gaussian._semantic_feature = clone_tensor(self._semantic_feature)
        new_gaussian.is_keygaussian = clone_tensor(self.is_keygaussian)
        new_gaussian.cluster_parent_gaussian_index = clone_tensor(self.cluster_parent_gaussian_index)
        # new_gaussian.key_gaussian_indices = clone_tensor(self.key_gaussian_indices)
        if self.initial_xyz != None:
            new_gaussian.initial_xyz = clone_tensor(self.initial_xyz)
        
        
        # Copy relative_positions and relative_rotations if they exist
        if hasattr(self, 'relative_positions') and hasattr(self, 'relative_rotations'):
            new_gaussian.relative_positions = self.relative_positions.clone().detach()
            new_gaussian.relative_rotations = self.relative_rotations.clone().detach()
        if hasattr(self, 'cluster_label') and self.cluster_label is not None:
            new_gaussian.cluster_label = clone_tensor(self.cluster_label)
        if hasattr(self, 'cluster_control_points') and self.cluster_control_points is not None:
            new_gaussian.cluster_control_points = clone_tensor(self.cluster_control_points)
        if hasattr(self, 'cluster_control_orientations') and self.cluster_control_orientations is not None:
            new_gaussian.cluster_control_orientations = clone_tensor(self.cluster_control_orientations)
        if hasattr(self, 'relative_positions') and self.relative_positions is not None:
            new_gaussian.relative_positions = clone_tensor(self.relative_positions)
        

        # Copy other necessary attributes
        new_gaussian.setup_functions()  # Ensure functions are set up in the new instance
        
        return new_gaussian



    def state_dict(self):
        """Returns a dictionary containing all relevant parameters to save."""
        return {
            'xyz': self._xyz.cpu().detach().numpy(),
            'features_dc': self._features_dc.cpu().detach().numpy(),
            'features_rest': self._features_rest.cpu().detach().numpy(),
            'scaling': self._scaling.cpu().detach().numpy(),
            'rotation': self._rotation.cpu().detach().numpy(),
            'opacity': self._opacity.cpu().detach().numpy(),
            'semantic_feature': self._semantic_feature.cpu().detach().numpy(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'cluster_label': self.cluster_label.cpu().detach().numpy() if self.cluster_label is not None else None,
            'relative_positions': self.relative_positions.cpu().detach().numpy() if hasattr(self, 'relative_positions') else None,
            'relative_rotations': self.relative_rotations.cpu().detach().numpy() if hasattr(self, 'relative_rotations') else None
        }

    def load_state_dict(self, state_dict):
        """Loads the Gaussian model parameters from a saved state_dict."""
        self._xyz = torch.tensor(state_dict['xyz'], device="cuda").requires_grad_(True)
        self._features_dc = torch.tensor(state_dict['features_dc'], device="cuda").requires_grad_(True)
        self._features_rest = torch.tensor(state_dict['features_rest'], device="cuda").requires_grad_(True)
        self._scaling = torch.tensor(state_dict['scaling'], device="cuda").requires_grad_(True)
        self._rotation = torch.tensor(state_dict['rotation'], device="cuda").requires_grad_(True)
        self._opacity = torch.tensor(state_dict['opacity'], device="cuda").requires_grad_(True)
        self._semantic_feature = torch.tensor(state_dict['semantic_feature'], device="cuda").requires_grad_(True)

        if state_dict.get('cluster_label') is not None:
            self.cluster_label = torch.tensor(state_dict['cluster_label'], device="cuda").long()

        if state_dict.get('relative_positions') is not None:
            self.relative_positions = torch.tensor(state_dict['relative_positions'], device="cuda").requires_grad_(True)

        if state_dict.get('relative_rotations') is not None:
            self.relative_rotations = torch.tensor(state_dict['relative_rotations'], device="cuda").requires_grad_(True)

        if self.optimizer and state_dict.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        print("Gaussian model state loaded.")




def quaternion_inverse(q):
    """
    Compute the inverse of a quaternion.

    Args:
        q (torch.Tensor): Tensor of shape (..., 4), representing quaternions (w, x, y, z).

    Returns:
        torch.Tensor: Inverse quaternions of the same shape as input.
    """
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
    return q_conj / norm_sq

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    Args:
        q1 (torch.Tensor): Tensor of shape (..., 4), representing quaternions (w, x, y, z).
        q2 (torch.Tensor): Tensor of shape (..., 4), representing quaternions (w, x, y, z).

    Returns:
        torch.Tensor: Resulting quaternions after multiplication, same shape as input.
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack((w, x, y, z), dim=-1)

def rotate_vector(q, v):
    """
    Rotate a vector by a quaternion.

    Args:
        q (torch.Tensor): Tensor of shape (..., 4), representing quaternions (w, x, y, z).
        v (torch.Tensor): Tensor of shape (..., 3), representing vectors.

    Returns:
        torch.Tensor: Rotated vectors of shape (..., 3).
    """
    # Convert vector to quaternion with zero scalar part
    zeros = torch.zeros_like(v[..., :1])
    v_quat = torch.cat([zeros, v], dim=-1)  # Shape (..., 4)
    
    # Compute q * v * q^-1
    q_inv = quaternion_inverse(q)
    qv = quaternion_multiply(q, v_quat)
    rotated_quat = quaternion_multiply(qv, q_inv)
    
    return rotated_quat[..., 1:]  # Extract vector part