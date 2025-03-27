# scene/blender_dataset_reader.py

# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import List
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model_inria import BasicPointCloud
import torch
from collections import defaultdict
import cv2

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import List

from typing import NamedTuple, Optional
from PIL import Image
import torch
import numpy as np


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FoVy: float
    FoVx: float    
    width: Optional[int] = None
    height: Optional[int] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    cameras: List[CameraInfo]      # List of CameraInfo objects, each containing multiple FrameInfos
    nerf_normalization: dict       # Normalization data for NeRF models
    ply_path: str                  # Path to point cloud .ply file
    semantic_feature_dim: int       # Dimension of semantic features


def getNerfppNorm(frame_infos):
    """
    TODO: What does this function do?
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for frame in frame_infos:
        R = frame.R
        T = frame.T
        W2C = getWorld2View2(R, T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def read_cameras_from_meta_json_videoloading(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".jpg"):
    cam_infos = []
    print(f"Reading cameras from {path} with {transformsfile}")
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # Retrieve the camera parameters from the JSON file
        width = contents["width"]
        height = contents["height"]

        # Extract the intrinsic matrix K
        K = np.array(contents["k"])[0]

        # Extract the focal lengths
        fx = K[0, 0]
        fy = K[1, 1]

        # Compute horizontal and vertical FoV in degrees
        FoVx = 2 * np.arctan(width / (2 * fx)) # * 180 / np.pi
        FoVy = 2 * np.arctan(height / (2 * fy))# * 180 / np.pi

        # Get camera world-to-camera matrices
        w2c_matrices = contents.get("w2c", [])
        c2w_matrices = contents.get("c2w", [])

        for idx, c2w_matrix in enumerate(c2w_matrices):
            cam_name = f"cam_{idx:03d}"  # Camera name with leading zeros

            # Convert w2c_matrix to numpy array
            c2w = np.array(c2w_matrix)
            
            # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = contents["img_path"][idx]
            image_name = Path(image_path).stem

            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping this image.")
                continue

            # Handle background and alpha channel
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray((arr * 255.0).astype(np.uint8), "RGB")


            video_path = contents["vid_path"][idx]
            video_name = Path(video_path).stem    
            video = cv2.VideoCapture(video_path)
            video_frames = []
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                video_frames.append(frame)
            video.release()


            # Create CameraInfo object
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.width,
                height=image.height,
                video=video,
                video_path=video_path,
                video_name=video_name,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_name=None,
                time=None
            )
            cam_infos.append(cam_info)

    return cam_infos


def read_cameraInfos_from_meta_json(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []
    print(f"Reading cameras from {os.path.basename(path)} with {transformsfile}")
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        width = contents["width"]
        height = contents["height"]

        K = np.array(contents["k"])[0]

        fx = K[0, 0]
        fy = K[1, 1]

        FoVx = 2 * np.arctan(width / (2 * fx)) # * 180 / np.pi
        FoVy = 2 * np.arctan(height / (2 * fy))# * 180 / np.pi

        w2c_matrices = contents.get("w2c", [])
        c2w_matrices = contents.get("c2w", [])

        for idx, c2w_matrix in enumerate(c2w_matrices):
            cam_name = f"cam_{idx:03d}"  # Camera name with leading zeros

            c2w = np.array(c2w_matrix)
            
            c2w[:3, 1:3] *= -1  # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Create CameraInfo object
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                width=width,
                height=height
            )
            cam_infos.append(cam_info)

    return cam_infos


def readSceneInfoBlender(path, white_background, foundation_model=None):
    """Read scene information from Blender dataset."""
    if foundation_model == 'sam':
        semantic_feature_dir = "sam_embeddings"
    elif foundation_model == 'lseg':
        semantic_feature_dir = "rgb_feature_langseg"
    else:
        semantic_feature_dir = ""

    use_semantic_features = bool(semantic_feature_dir)

    print("Reading cam_infos from train_meta.json")
    cam_infos = read_cameraInfos_from_meta_json(path, "train_meta.json", white_background)

    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 6 - 3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # Set semantic feature dimension to 0 since we're not using semantic features in this version
    semantic_feature_dim = 0

    scene_info = SceneInfo(
        point_cloud=pcd,
        cameras=cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        semantic_feature_dim=semantic_feature_dim
    )
    return scene_info

