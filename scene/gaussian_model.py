# gaussian_model.py:

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
from sklearn.neighbors import NearestNeighbors

from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import copy
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
#from pointnet.semantic_segmentation.pointnet2_utils import farthest_point_sample
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



def compute_normals(points_np, k=10):
    """
    Compute normals for a point cloud (numpy array of shape (N,3)) using a simple PCA on local neighborhoods.
    Returns an array of shape (N,3) with normals.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points_np)
    distances, indices = nbrs.kneighbors(points_np)
    normals = []
    for i in range(points_np.shape[0]):
        # Exclude the point itself (first neighbor)
        neighbors = points_np[indices[i][1:]]
        cov = np.cov(neighbors - points_np[i], rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # eigenvector with smallest eigenvalue
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        normals.append(normal)
    return np.array(normals)




class GaussianModel:



    def __init__(self, sh_degree: int, device: str = 'cuda'):
        
        self.device = device

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        # "Trainable" parameters
        self._xyz               = nn.Parameter(torch.empty(0, 3, device=self.device, requires_grad=True))
        self._features_dc       = nn.Parameter(torch.empty(0, 1, 3, device=self.device, requires_grad=False))
        self._features_rest     = nn.Parameter(torch.empty(0, 15, 3, device=self.device, requires_grad=False))
        self._scaling           = nn.Parameter(torch.empty(0, 3, device=self.device, requires_grad=False))
        self._rotation          = nn.Parameter(torch.empty(0, 4, device=self.device, requires_grad=False))
        self._opacity           = nn.Parameter(torch.empty(0, 1, device=self.device, requires_grad=False))
        self._semantic_feature  = nn.Parameter(torch.empty(0, 1, 1, device=self.device, requires_grad=False))

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
        self.xyz_cp = None              # Tensor of shape [n_clusters, 3] (xyz positions of control points)
        self.rot_cp = None              # Tensor of shape [n_clusters, 4] (initially identity) (quaternions of control points)
        self.xyz_rel = None             # For each gaussian: offset from its cluster control point.
        self.rot_rel = None             # For each gaussian: relative rotation with respect to its cluster control point.

        self.semantic_class_color = None



    def cluster_gaussians_old(self, n_clusters=3):
        """
        Cluster gaussians into 3 clusters based solely on their 3D positions.
        For each cluster the control point is defined as the mean of the positions
        of all gaussians in that cluster. The control orientation is initially the identity quaternion.
        Also, for each gaussian, compute and store its relative offset with respect to its cluster control point.
        """
        positions = self.get_xyz.detach().cpu().numpy()  # (N, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(positions)
        labels = kmeans.labels_  # (N,)
        self.cluster_label = torch.tensor(labels, device=self.device, dtype=torch.long).unsqueeze(1)

        # Compute cluster control points (mean of positions in each cluster)
        xyz_cp = []
        for cl in range(n_clusters):
            indices = np.where(labels == cl)[0]
            cluster_mean = np.mean(positions[indices], axis=0)
            xyz_cp.append(cluster_mean)
        self.xyz_cp = torch.tensor(xyz_cp, device=self.device, dtype=self.get_xyz.dtype)

        # Set control orientations to identity for each cluster.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.get_xyz.dtype)
        self.rot_cp = identity_quat.repeat(n_clusters, 1)

        # Compute relative offsets: for each gaussian, relative offset = gaussian position - (cluster control point)
        N = positions.shape[0]
        rel_positions = []
        for i in range(N):
            cl = labels[i]
            rel = positions[i] - self.xyz_cp[cl].cpu().numpy()
            rel_positions.append(rel)
        self.xyz_rel = torch.tensor(rel_positions, device=self.device, dtype=self.get_xyz.dtype)


    def initialize_controlpoints(self, num_objects):
        """
        Cluster all Gaussians into num_objects clusters, and initialize control points.
        
        For each cluster:
        - Set control_point.position as the mean of the positions of all Gaussians in that cluster.
        - Set control_point.quaternion as the identity unit quaternion.
        - For each Gaussian in the cluster, compute:
                relative_position = gaussian.position - control_point.position
                relative_quaternion = quaternion_multiply(quaternion_inverse(control_point.quaternion), gaussian.quaternion)
        
        This function updates the following attributes:
        - self.cluster_label (Tensor of shape [N, 1])
        - self.xyz_cp (Tensor of shape [num_objects, 3])
        - self.rot_cp (Tensor of shape [num_objects, 4])
        - self.xyz_rel (Tensor of shape [N, 3])
        - self.rot_rel (Tensor of shape [N, 4])
        """

        # 1. Cluster the Gaussians based on their positions.
        positions = self.get_xyz.detach().cpu().numpy()  # shape (N, 3)
        kmeans = KMeans(n_clusters=num_objects, random_state=42).fit(positions)
        labels = kmeans.labels_  # shape (N,)
        
        # Store cluster labels (as a tensor of shape [N, 1])
        self.cluster_label = torch.tensor(labels, device=self.device, dtype=torch.long).unsqueeze(1)

        # 2. Compute cluster control points (mean positions) and assign identity quaternions.
        xyz_cp = []
        for i in range(num_objects):
            indices = np.where(labels == i)[0]
            if indices.size == 0:
                # If a cluster happens to be empty, set a default value (e.g., origin)
                cluster_mean = np.array([0.0, 0.0, 0.0])
                raise ValueError(f"Cluster {i} is empty - this should never happen with K-means clustering")
            else:
                cluster_mean = positions[indices].mean(axis=0)
            xyz_cp.append(cluster_mean)
        self.xyz_cp = torch.tensor(xyz_cp, device=self.device, dtype=self.get_xyz.dtype)
        
        # Use identity quaternions for all control points: (1, 0, 0, 0)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.get_xyz.dtype)
        self.rot_cp = identity_quat.repeat(num_objects, 1)
        
        # 3. Compute relative positions and relative rotations for each Gaussian.
        # relative_position = gaussian.position - cluster_control_point
        # relative_rotation = quaternion_inverse(control_quat) * gaussian.quaternion
        gaussian_positions = self.get_xyz  # shape (N, 3)
        gaussian_rotations = self.get_rotation  # shape (N, 4)
        
        # Vectorized computation of relative positions
        # Use advanced indexing to get control points for each gaussian
        control_pts = self.xyz_cp[labels]  # Shape: [N, 3]
        self.xyz_rel = gaussian_positions - control_pts  # Shape: [N, 3]
        
        # Vectorized computation of relative rotations
        control_quats = self.rot_cp[labels]  # Shape: [N, 4]
        inv_control_quats = quaternion_inverse(control_quats)  # Shape: [N, 4]
        self.rot_rel = quaternion_multiply(inv_control_quats, gaussian_rotations)  # Shape: [N, 4]
        

    def run_pointnet_semantic_part_segmentation(self, segmentation_model, num_sample_points=1024, num_classes=50, knn_k=10, overwrite_colors=True):
        """
        Run semantic part segmentation on the gaussians using a pretrained PointNet++ segmentation model.
        
        Steps:
        1. Subsample to a fixed number of gaussians (using farthest point sampling).
        2. Construct a point cloud with 9 features per point (xyz, rgb, normals).
        3. Pass the point cloud (reshaped to [B,9,N]) to the segmentation model.
        4. Obtain predicted labels for the down-sampled gaussians.
        5. For each full gaussian, assign the label of its closest down-sampled gaussian.
        6. Optionally, overwrite the gaussian colors with predicted semantic label colors.
        
        Returns:
            predicted_labels_full (Tensor of shape [N_total]) containing segmentation class labels.
        """
        device = self.get_xyz.device

        # 1. Subsample using farthest point sampling.
        full_xyz = self.get_xyz  # (N_total, 3)
        N_total = full_xyz.shape[0]
        if N_total < num_sample_points:
            raise ValueError(f"Number of gaussians ({N_total}) is less than the required sample count ({num_sample_points}).")
        
        full_xyz_unsq = full_xyz.unsqueeze(0)  # (1, N_total, 3)
        sample_indices = farthest_point_sample(full_xyz_unsq, num_sample_points)  # (1, num_sample_points)
        sample_indices = sample_indices.squeeze(0)  # (num_sample_points,)
        downsampled_xyz = full_xyz[sample_indices]  # (num_sample_points, 3)
        
        # 2. Construct the 9-feature point cloud.
        colors = self.colors.to(device).float() / 255.0  # (N_total, 3)
        downsampled_rgb = colors[sample_indices]  # (num_sample_points, 3)
        
        downsampled_xyz_np = downsampled_xyz.detach().cpu().numpy()
        normals_np = compute_normals(downsampled_xyz_np, k=knn_k)  # (num_sample_points, 3)
        downsampled_normals = torch.tensor(normals_np, device=device, dtype=downsampled_xyz.dtype)
        
        point_features = torch.cat([downsampled_xyz, downsampled_rgb, downsampled_normals], dim=1)
        point_features = point_features.transpose(0, 1).unsqueeze(0)  # (1, 9, num_sample_points)
        
        # 3. Run the segmentation model.
        segmentation_model = segmentation_model.to(device)
        segmentation_model.eval()
        with torch.no_grad():
            pred, _ = segmentation_model(point_features)
        predicted_labels_down = torch.argmax(pred, dim=2).squeeze(0)  # (num_sample_points,)
        
        # 4. For each full gaussian, assign the label of its closest down-sampled gaussian.
        dists = torch.cdist(full_xyz, downsampled_xyz)  # (N_total, num_sample_points)
        nearest_idx = torch.argmin(dists, dim=1)  # (N_total,)
        predicted_labels_full = predicted_labels_down[nearest_idx]  # (N_total,)
        
        self.semantic_labels = predicted_labels_full  # (N_total,)
        
        # 5. Optionally, overwrite the gaussian colors with predicted semantic label colors.
        if overwrite_colors:
            # Create a color mapping for each class.
            color_mapping_np = get_cluster_colors(num_classes)  # shape (num_classes, 3), uint8
            color_mapping_tensor = torch.tensor(color_mapping_np, device=device, dtype=torch.uint8)
            new_colors = color_mapping_tensor[predicted_labels_full]  # (N_total, 3)
            new_colors_float = new_colors.float() / 255.0  # normalized [0,1]
            # NEW: Instead of modifying _features_dc and _features_rest, simply store the semantic class colors.
            self.semantic_class_color = new_colors_float.clone()
            # Optionally, also disable SH-based coloring.
            self.active_sh_degree = None
        
        return predicted_labels_full


    def cluster_gaussians_with_semantic_features(self, num_clusters, overwrite_colors=True):
        """
        Cluster gaussians using their positions and semantic features, and assign cluster colors.
        
        This method assumes that:
        - self.get_xyz returns a tensor of shape (N, 3) containing gaussian positions.
        - self.semantic_labels is available (e.g. after run_pointnet_semantic_part_segmentation)
            and is a tensor of shape (N,) with integer semantic labels.
        
        The semantic labels are converted to one‑hot vectors (assuming 13 semantic classes, adjust if needed)
        and concatenated with the positions to form a feature vector of dimension 16.
        KMeans is then used to cluster the gaussians into num_clusters clusters.
        
        If overwrite_colors is True, this method will assign a cluster color to each gaussian by
        computing distinct colors (using get_cluster_colors) for each cluster.
        
        Args:
            num_clusters (int): Desired number of clusters.
            overwrite_colors (bool): If True, sets self.cluster_color with per-gaussian RGB colors.
        
        Returns:
            torch.Tensor: A tensor of shape (N,) containing the cluster label for each gaussian.
        """
        # 1. Get positions as a numpy array, shape (N, 3)
        positions = self.get_xyz.cpu().detach().numpy()
        
        # 2. Ensure semantic labels are available.
        if not hasattr(self, 'semantic_labels'):
            raise ValueError("Semantic labels not found. Please run run_pointnet_semantic_part_segmentation first.")
        semantic_labels = self.semantic_labels.cpu().detach().numpy()  # shape (N,)
        
        # 3. Convert semantic labels to one-hot vectors.
        num_semantic_classes = 13  # Adjust if needed.
        semantic_onehot = np.eye(num_semantic_classes)[semantic_labels]  # shape (N, 13)
        
        # 4. Concatenate positions and semantic one-hot features to form feature vectors.
        features = np.concatenate([positions, semantic_onehot], axis=1)  # shape (N, 16)
        
        # 5. Cluster using KMeans.
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
        cluster_labels = kmeans.labels_  # shape (N,)
        
        # Store the cluster labels as a tensor.
        self.semantic_cluster_labels = torch.tensor(cluster_labels, device=self.device, dtype=torch.long)
        
        # 6. Optionally assign a distinct color to each cluster.
        if overwrite_colors:
            # get_cluster_colors returns an array of shape (num_clusters, 3) with RGB values in [0, 255]
            cluster_colors_np = get_cluster_colors(num_clusters)
            # Normalize colors from [0, 255] to [0, 1]
            cluster_colors = torch.tensor(cluster_colors_np, device=self.device, dtype=torch.float32) / 255.0
            # For each gaussian, assign the color of its cluster.
            self.cluster_color = cluster_colors[self.semantic_cluster_labels]  # shape (N, 3)
            # Also optionally store the list of individual cluster colors.
            self.individual_cluster_colors = cluster_colors

        print(f"Clustered {features.shape[0]} gaussians into {num_clusters} clusters.")
        return self.semantic_cluster_labels




    def initialize_controlpoints_onehotencoding(self, num_objects):
        """
        Cluster all Gaussians into num_objects clusters, and initialize control points.
        
        For each cluster:
        - Set control_point.position as the mean of the positions of all Gaussians in that cluster.
        - Set control_point.quaternion as the identity unit quaternion.
        - For each Gaussian in the cluster, compute:
                relative_position = gaussian.position - control_point.position
                relative_quaternion = quaternion_multiply(quaternion_inverse(control_point.quaternion), gaussian.quaternion)
        
        This function updates the following attributes:
        - self.cluster_label (One-hot Tensor of shape [N, num_objects])
        - self.xyz_cp (Tensor of shape [num_objects, 3])
        - self.rot_cp (Tensor of shape [num_objects, 4])
        - self.xyz_rel (Tensor of shape [N, 3])
        - self.rot_rel (Tensor of shape [N, 4])
        """

        # 1. Cluster the Gaussians based on their positions.
        positions = self.get_xyz.detach().cpu().numpy()  # shape (N, 3)
        kmeans = KMeans(n_clusters=num_objects, random_state=42).fit(positions)
        labels = kmeans.labels_  # shape (N,)

        # Convert integer labels to one-hot encoding
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.long)
        onehot = torch.nn.functional.one_hot(labels_tensor, num_classes=num_objects).float()  # shape (N, num_objects)
        self.cluster_label = onehot

        # 2. Compute cluster control points (mean positions) and assign identity quaternions.
        xyz_cp = []
        for i in range(num_objects):
            indices = np.where(labels == i)[0]
            if indices.size == 0:
                raise ValueError(f"Cluster {i} is empty - this should never happen with K-means clustering")
            else:
                cluster_mean = positions[indices].mean(axis=0)
            xyz_cp.append(cluster_mean)
        self.xyz_cp = torch.tensor(xyz_cp, device=self.device, dtype=self.get_xyz.dtype)
        
        # Use identity quaternions for all control points: (1, 0, 0, 0)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.get_xyz.dtype)
        self.rot_cp = identity_quat.repeat(num_objects, 1)
        
        # 3. Compute relative positions and relative rotations for each Gaussian.
        # Instead of using integer indexing, compute the control point per Gaussian as a weighted sum.
        gaussian_positions = self.get_xyz  # shape (N, 3)
        gaussian_rotations = self.get_rotation  # shape (N, 4)
        
        # Compute control points for each Gaussian by matrix multiplying the one-hot labels with control points.
        control_pts = onehot @ self.xyz_cp  # shape: (N, 3)
        self.xyz_rel = gaussian_positions - control_pts  # shape: (N, 3)
        
        # Similarly, compute control quaternions per Gaussian
        control_quats = onehot @ self.rot_cp  # shape: (N, 4)
        inv_control_quats = quaternion_inverse(control_quats)  # shape: (N, 4)
        self.rot_rel = quaternion_multiply(inv_control_quats, gaussian_rotations)  # shape: (N, 4)



    def update_gaussians_from_controlpoints(self):
        """
        Update each child Gaussian's absolute transformation based on its parent's updated control point.
        
        For each Gaussian:
        - The new position is computed as:
            new_position = control_point_position + rotate_vector(control_point_quaternion, relative_position)
        - The new rotation is computed as:
            new_rotation = quaternion_multiply(control_point_quaternion, relative_rotation)
        
        This function expects that the following attributes are set:
        - self.cluster_label: Tensor of shape [N, 1] with the cluster (control point) index for each Gaussian.
        - self.xyz_cp: Tensor of shape [num_objects, 3] containing updated control point positions.
        - self.rot_cp: Tensor of shape [num_objects, 4] containing updated control point quaternions.
        - self.xyz_rel: Tensor of shape [N, 3] storing each Gaussian's offset (in its parent's local coordinates).
        - self.rot_rel: Tensor of shape [N, 4] storing each Gaussian's relative rotation.
        
        After computing the new transformations, this function updates the internal parameters:
        - self._xyz (positions)
        - self._rotation (rotations)
        """
        # Ensure that the cluster labels are a 1D tensor of indices.
        labels = self.cluster_label.squeeze()  # shape: (N,)
        
        # Gather the corresponding control point positions and orientations for each Gaussian.
        control_positions = self.xyz_cp[labels]          # shape: (N, 3)
        control_orientations = self.rot_cp[labels]    # shape: (N, 4)
        
        # Rotate the stored relative positions by the control point's quaternion.
        # The helper function rotate_vector expects the quaternion and a vector.
        rotated_rel_positions = rotate_vector(control_orientations, self.xyz_rel)  # shape: (N, 3)
        
        # Update absolute positions.
        new_positions = control_positions + rotated_rel_positions
        
        # Update absolute rotations.
        new_rotations = quaternion_multiply(control_orientations, self.rot_rel)
        
        # Update the internal Gaussian parameters.
        self._xyz = new_positions.to(self.device)
        self._rotation = new_rotations.to(self.device)


    def update_gaussians(self, new_xyz, new_rot, scaling_factor=1.0):

        self._xyz = new_xyz.to(self.device)
        self._rotation = new_rot.to(self.device)
        self._scaling = self._scaling*scaling_factor

        # labels = self.cluster_label.squeeze()  # shape: (N,)
        
        # # Gather the corresponding control point positions and orientations
        # control_positions = self.xyz_cp[labels]          # shape: (N, 3)
        # control_orientations = self.rot_cp[labels]       # shape: (N, 4)
        
        # # Rotate the stored relative positions
        # rotated_rel_positions = rotate_vector(control_orientations, self.xyz_rel)
        
        # # Update absolute positions
        # new_positions = control_positions + rotated_rel_positions
        
        # # Update absolute rotations
        # new_rotations = quaternion_multiply(control_orientations, self.rot_rel)
        
        # # Update the internal Gaussian parameters
        # self._xyz = new_positions.to(self.device)
        # self._rotation = new_rotations.to(self.device)


    def setup_functions(self):
        
        self.scaling_activation             = torch.exp
        self.scaling_inverse_activation     = torch.log
        self.covariance_activation          = build_covariance_from_scaling_rotation
        self.opacity_activation             = torch.sigmoid
        self.inverse_opacity_activation     = inverse_sigmoid
        self.rotation_activation            = torch.nn.functional.normalize

    def update_from_checkpoint(self, checkpoint):
        """Update Gaussian model parameters directly from a checkpoint without using Scene."""
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
         self._semantic_feature) = checkpoint


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
            self._semantic_feature

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
        self._semantic_feature) = model_args
        self.training_setup_0(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def restore_msgnode(self, model_args):
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
        self._semantic_feature) = model_args


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
        

    def update_dynamic_state(self, gt_xyz, gt_rot):
        """
        Update the dynamic parameters (positions and rotations) from precomputed tensors.
        
        Args:
            t_index (int): Time index (0 <= t_index < T_dynamic).
            gt_xyz (torch.Tensor): Tensor of shape [T_dynamic, N, 3] with dynamic positions.
            gt_rot (torch.Tensor): Tensor of shape [T_dynamic, N, 4] with dynamic rotations.
        """
    
        self._xyz = gt_xyz
        self._rotation = gt_rot


    def sparsify(self, sparsify_factor: float):
        """
        Randomly keep a fraction of the gaussians and delete the rest.
        
        Args:
            sparsify_factor (float): Fraction of gaussians to keep (e.g., 0.3 means keep 30%).
        """
        # Get total number of gaussians
        num_points = self.get_xyz.shape[0]
        # Calculate number of points to keep
        num_keep = int(sparsify_factor * num_points)
        
        if num_keep <= 0 or num_keep > num_points:
            raise ValueError("sparsify_factor must be in the range (0, 1].")
        
        # Create a random permutation of indices and select the first num_keep indices
        perm = torch.randperm(num_points, device=self.device)
        keep_indices = perm[:num_keep]
        
        # Create a boolean mask where True indicates the gaussian should be deleted.
        # Mark all indices as True (to delete), then set the kept indices to False.
        mask = torch.ones(num_points, dtype=torch.bool, device=self.device)
        mask[keep_indices] = False
        
        # Use the existing prune_points method to remove the gaussians
        self.prune_points(mask)


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
        self.xyz_scheduler_args = get_expon_lr_func(lr_init         = training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final        = training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult   = training_args.position_lr_delay_mult,
                                                    max_steps       = training_args.position_lr_max_steps)

    def training_setup_t_old(self, training_args):
        self._xyz.requires_grad = True
        self._rotation.requires_grad = True
        self._opacity.requires_grad = False
        self._scaling.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._semantic_feature.requires_grad = False

        # l = [
        #     {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        #     {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        # ]
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init         = training_args.position_lr_init * self.spatial_lr_scale,
        #                                             lr_final        = training_args.position_lr_final * self.spatial_lr_scale,
        #                                             lr_delay_mult   = training_args.position_lr_delay_mult,
        #                                             max_steps       = training_args.position_lr_max_steps)

    def training_setup_arap(self, training_args):
        self._xyz = nn.Parameter(self._xyz, requires_grad=True)
        self._rotation = nn.Parameter(self._rotation, requires_grad=True)
        self._opacity = nn.Parameter(self._opacity, requires_grad=False)
        self._scaling = nn.Parameter(self._scaling, requires_grad=False)
        self._features_dc = nn.Parameter(self._features_dc, requires_grad=False)
        self._features_rest = nn.Parameter(self._features_rest, requires_grad=False)
        self._semantic_feature = nn.Parameter(self._semantic_feature, requires_grad=False)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init         = training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final        = training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult   = training_args.position_lr_delay_mult,
                                                    max_steps       = training_args.position_lr_max_steps)


    def training_setup_t(self, training_args):
        self._xyz = nn.Parameter(self._xyz.detach(), requires_grad=True)
        self._rotation = nn.Parameter(self._rotation.detach(), requires_grad=True)
        self._opacity = nn.Parameter(self._opacity.detach(), requires_grad=False)
        self._scaling = nn.Parameter(self._scaling.detach(), requires_grad=False)
        self._features_dc = nn.Parameter(self._features_dc.detach(), requires_grad=False)
        self._features_rest = nn.Parameter(self._features_rest.detach(), requires_grad=False)
        self._semantic_feature = nn.Parameter(self._semantic_feature.detach(), requires_grad=False)

        if self.optimizer is None:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            

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
        if self.optimizer is None:
        # Directly prune internal tensors without touching the optimizer state.
            return {
                "xyz": self._xyz[mask],
                "f_dc": self._features_dc[mask],
                "f_rest": self._features_rest[mask],
                "opacity": self._opacity[mask],
                "scaling": self._scaling[mask],
                "rotation": self._rotation[mask],
                "semantic_feature": self._semantic_feature[mask]
            }
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
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[0,update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

        
    def clone(self):
        new_gaussian = GaussianModel(self.max_sh_degree)
        new_gaussian.active_sh_degree = self.active_sh_degree
        new_gaussian.spatial_lr_scale = self.spatial_lr_scale

        # Helper to clone and detach tensors so they become leaf nodes.
        def clone_tensor(tensor):
            if isinstance(tensor, nn.Parameter):
                return nn.Parameter(tensor.detach().clone(), requires_grad=tensor.requires_grad)
            else:
                return tensor.detach().clone()
        
        new_gaussian.cluster_label = clone_tensor(self.cluster_label) if self.cluster_label is not None else None
        new_gaussian._xyz = clone_tensor(self._xyz)
        new_gaussian._features_dc = clone_tensor(self._features_dc)
        new_gaussian._features_rest = clone_tensor(self._features_rest)
        new_gaussian._scaling = clone_tensor(self._scaling)
        new_gaussian._rotation = clone_tensor(self._rotation)
        new_gaussian._opacity = clone_tensor(self._opacity)
        new_gaussian.max_radii2D = clone_tensor(self.max_radii2D)
        new_gaussian.xyz_gradient_accum = clone_tensor(self.xyz_gradient_accum)
        new_gaussian.denom = clone_tensor(self.denom)
        new_gaussian._semantic_feature = clone_tensor(self._semantic_feature)

        if self.initial_xyz is not None:
            new_gaussian.initial_xyz = clone_tensor(self.initial_xyz)
        
        if hasattr(self, 'xyz_rel') and self.xyz_rel is not None:
            new_gaussian.xyz_rel = clone_tensor(self.xyz_rel)
        
        if hasattr(self, 'rot_rel') and self.rot_rel is not None:
            new_gaussian.rot_rel = clone_tensor(self.rot_rel)
        
        if hasattr(self, 'cluster_label') and self.cluster_label is not None:
            new_gaussian.cluster_label = clone_tensor(self.cluster_label)
        
        if hasattr(self, 'xyz_cp') and self.xyz_cp is not None:
            new_gaussian.xyz_cp = clone_tensor(self.xyz_cp)
        
        if hasattr(self, 'rot_cp') and self.rot_cp is not None:
            new_gaussian.rot_cp = clone_tensor(self.rot_cp)
        
        # Copy other necessary attributes and setup functions
        new_gaussian.setup_functions()
        
        return new_gaussian



    def clone_old(self):
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


        if self.initial_xyz != None:
            new_gaussian.initial_xyz = clone_tensor(self.initial_xyz)
        
        

        if hasattr(self, 'xyz_rel') and self.xyz_rel is not None:
            new_gaussian.xyz_rel = self.xyz_rel.clone().detach()

        if hasattr(self, 'rot_rel') and self.rot_rel is not None:
            new_gaussian.rot_rel = self.rot_rel.clone().detach()

        if hasattr(self, 'cluster_label') and self.cluster_label is not None:
            new_gaussian.cluster_label = clone_tensor(self.cluster_label)

        if hasattr(self, 'xyz_cp') and self.xyz_cp is not None:
            new_gaussian.xyz_cp = clone_tensor(self.xyz_cp)

        if hasattr(self, 'rot_cp') and self.rot_cp is not None:
            new_gaussian.rot_cp = clone_tensor(self.rot_cp)

        

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
            'xyz_rel': self.xyz_rel.cpu().detach().numpy() if hasattr(self, 'xyz_rel') else None,
            'rot_rel': self.rot_rel.cpu().detach().numpy() if hasattr(self, 'rot_rel') else None
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

        if state_dict.get('xyz_rel') is not None:
            self.xyz_rel = torch.tensor(state_dict['xyz_rel'], device="cuda").requires_grad_(True)

        if state_dict.get('rot_rel') is not None:
            self.rot_rel = torch.tensor(state_dict['rot_rel'], device="cuda").requires_grad_(True)

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