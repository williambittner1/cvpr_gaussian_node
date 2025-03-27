# scene/dataset_readers.py

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

class CameraInfo_video(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FoVy: float
    FoVx: float
    
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    image_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    video: Optional[np.ndarray] = None
    video_path: Optional[str] = None
    video_name: Optional[str] = None

    semantic_feature: Optional[torch.Tensor] = None
    semantic_feature_path: Optional[str] = None
    semantic_feature_name: Optional[str] = None
    time: Optional[int] = None


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FoVy: float
    FoVx: float
    
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    image_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    semantic_feature: Optional[torch.Tensor] = None
    semantic_feature_path: Optional[str] = None
    semantic_feature_name: Optional[str] = None
    time: Optional[int] = None




# class CameraInfo(NamedTuple):
#     uid: int           # Unique identifier for the camera
#     R: np.ndarray      # Rotation matrix (3x3 numpy array)
#     T: np.ndarray      # Translation vector (3-element numpy array)
#     FoVy: float        # Vertical field of view in degrees
#     FoVx: float        # Horizontal field of view in degrees
#     frames: List['FrameInfo']   # List of frames captured by this camera
#     semantic_features: List[torch.Tensor] = None  # Optional list of semantic features

    


class FrameInfo(NamedTuple):
    uid: int                      # Unique identifier for the frame
    image: Image.Image            # The image data (PIL Image)
    image_path: str               # Path to the image file
    image_name: str               # Name of the image file (without extension)
    width: int                    # Width of the image
    height: int                   # Height of the image
    semantic_feature: torch.Tensor         # The semantic feature tensor
    semantic_feature_path: str             # Path to the semantic feature file
    semantic_feature_name: str             # Name of the semantic feature file (without extension)
    time: int                      # Time attribute (optional)
    

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    cameras: List[CameraInfo]      # List of CameraInfo objects, each containing multiple FrameInfos
    nerf_normalization: dict       # Normalization data for NeRF models
    ply_path: str                  # Path to point cloud .ply file
    semantic_feature_dim: int      # Dimension of the semantic feature



def getNerfppNorm(frame_infos):
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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_root_folder, semantic_features_root_folder):
    from collections import defaultdict

    camera_infos = {}  # Key: camera_name, Value: CameraInfo
    uid_counter = 0    # To assign unique IDs to frames

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write(f"\rProcessing camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        # Extract camera index from extr.name
        basename = os.path.splitext(os.path.basename(extr.name))[0]
        tokens = basename.split('_')
        if len(tokens) < 3 or tokens[0] != 'cam':
            print(f"\nInvalid image name format: {extr.name}. Skipping.")
            continue
        camera_index = int(tokens[1])  # '01' -> 1
        camera_name = f"cam_{camera_index:02d}"

        # If CameraInfo already exists, skip creation
        if camera_name in camera_infos:
            camera_info = camera_infos[camera_name]
        else:
            # Create CameraInfo
            R = np.transpose(qvec2rotmat(extr.qvec))  # Rotation matrix
            T = np.array(extr.tvec)                   # Translation vector

            # Calculate Field of View once per camera
            if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                focal_length_x = intr.params[0]
                FoVy = focal2fov(focal_length_x, height)
                FoVx = focal2fov(focal_length_x, width)
            elif intr.model in ["PINHOLE", "OPENCV"]:
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FoVy = focal2fov(focal_length_y, height)
                FoVx = focal2fov(focal_length_x, width)
            else:
                raise ValueError(f"Unsupported Colmap camera model: {intr.model}")

            camera_info = CameraInfo(
                uid=intr.id,
                R=R,
                T=T,
                FoVy=float(FoVy),
                FoVx=float(FoVx),
                frames=[]  # Initialize empty list for frames
            )
            camera_infos[camera_name] = camera_info

        # Define paths for images and semantic features
        image_folder = os.path.join(images_root_folder, camera_name)
        semantic_feature_folder = os.path.join(semantic_features_root_folder, camera_name)

        # Check if directories exist
        if not os.path.isdir(image_folder) or not os.path.isdir(semantic_feature_folder):
            print(f"\nRequired folders not found for camera {camera_name}. Skipping.")
            continue

        # List and sort image files and semantic feature files
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        semantic_feature_files = sorted([f for f in os.listdir(semantic_feature_folder) if f.lower().endswith('.pt')])

        if len(image_files) != len(semantic_feature_files):
            print(f"\nMismatch in number of images and semantic features for camera {camera_name}. Skipping.")
            continue

        # Create FrameInfo objects for each image
        for img_file, feat_file in zip(image_files, semantic_feature_files):
            img_path = os.path.join(image_folder, img_file)
            feat_path = os.path.join(semantic_feature_folder, feat_file)

            try:
                image = Image.open(img_path).convert("RGB")
                semantic_feature = torch.load(feat_path)
            except Exception as e:
                print(f"\nError loading data for {img_file}: {e}. Skipping this frame.")
                continue

            time_str = img_file.split('_')[1]
            time = int(time_str)

            frame_info = FrameInfo(
                uid=uid_counter,
                image=image,
                image_path=img_path,
                image_name=os.path.splitext(img_file)[0],
                width=width,
                height=height,
                semantic_feature=semantic_feature,
                semantic_feature_path=feat_path,
                semantic_feature_name=os.path.splitext(feat_file)[0],
                time=time
            )
            camera_info.frames.append(frame_info)
            uid_counter += 1

    sys.stdout.write('\n')
    return list(camera_infos.values())  # Return list of CameraInfo objects



def readColmapCameras2(cam_extrinsics, cam_intrinsics, images_root_folder, semantic_features_root_folder):
    from collections import defaultdict

    camera_infos = {}  # Key: camera_name, Value: CameraInfo
    frame_infos = []   # List of FrameInfo objects
    uid_counter = 0    # To assign unique IDs to frames

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write(f"Processing camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        # Extract camera index from extr.name
        basename = os.path.splitext(os.path.basename(extr.name))[0]  # e.g., 'cam_01_001'
        tokens = basename.split('_')
        if len(tokens) < 3 or tokens[0] != 'cam':
            print(f"\nInvalid image name format: {extr.name}. Skipping.")
            continue
        camera_index = int(tokens[1])  # '01' -> 1

        camera_name = f"cam_{camera_index:02d}"

        # If CameraInfo already exists, skip creation
        if camera_name in camera_infos:
            camera_info = camera_infos[camera_name]
        else:
            # Create CameraInfo
            R = np.transpose(qvec2rotmat(extr.qvec))  # Rotation matrix
            T = np.array(extr.tvec)                   # Translation vector

            # Calculate Field of View once per camera
            if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                focal_length_x = intr.params[0]
                FoVy = focal2fov(focal_length_x, height)
                FoVx = focal2fov(focal_length_x, width)
            elif intr.model in ["PINHOLE", "OPENCV"]:
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FoVy = focal2fov(focal_length_y, height)
                FoVx = focal2fov(focal_length_x, width)
            else:
                raise ValueError(f"Unsupported Colmap camera model: {intr.model}")

            camera_info = CameraInfo(
                uid=intr.id,
                R=R,
                T=T,
                FoVy=float(FoVy),
                FoVx=float(FoVx),
            )
            camera_infos[camera_name] = camera_info

        # Define paths for images and semantic features
        image_folder = os.path.join(images_root_folder, camera_name)
        semantic_feature_folder = os.path.join(semantic_features_root_folder, camera_name)

        # Check if directories exist
        if not os.path.isdir(image_folder):
            print(f"\nImage folder {image_folder} does not exist. Skipping camera {camera_name}.")
            continue
        if not os.path.isdir(semantic_feature_folder):
            print(f"\nSemantic feature folder {semantic_feature_folder} does not exist. Skipping camera {camera_name}.")
            continue

        # List and sort image files and semantic feature files
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        semantic_feature_files = sorted([f for f in os.listdir(semantic_feature_folder) if f.lower().endswith('.pt')])

        # Ensure that the number of images and semantic features match
        if len(image_files) != len(semantic_feature_files):
            print(f"\nMismatch in number of images and semantic features for camera {camera_name}. Skipping.")
            continue

        # Create FrameInfo objects for each image
        for img_file, feat_file in zip(image_files, semantic_feature_files):
            img_path = os.path.join(image_folder, img_file)
            feat_path = os.path.join(semantic_feature_folder, feat_file)

            # Load image
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"\nError loading image {img_path}: {e}. Skipping this image.")
                continue

            # Load semantic feature
            try:
                semantic_feature = torch.load(feat_path)
            except Exception as e:
                print(f"\nError loading semantic feature {feat_path}: {e}. Skipping this feature.")
                continue
            
            # Extract time index from filename
            time_str = img_file.split('_')[1]  # Assuming filenames like 'cam1_001.jpg'
            time = int(time_str)
            
            frame_info = FrameInfo(
                camera_info=camera_info,
                uid=uid_counter,
                image=image,
                image_path=img_path,
                image_name=os.path.splitext(img_file)[0],
                width=width,
                height=height,
                semantic_feature=semantic_feature,
                semantic_feature_path=feat_path,
                semantic_feature_name=os.path.splitext(feat_file)[0],
                time=time  # Include time

            )
            frame_infos.append(frame_info)
            uid_counter += 1

    sys.stdout.write('\n')
    return frame_infos



def readColmapCameras_cam_info_with_multi_images(cam_extrinsics, cam_intrinsics, images_root_folder, semantic_features_root_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write(f"Reading camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))  # Rotation matrix
        T = np.array(extr.tvec)                   # Translation vector

        # Extract camera index and frame index from extr.name
        basename = os.path.splitext(os.path.basename(extr.name))[0]  # e.g., 'cam_01_001'
        tokens = basename.split('_')
        if len(tokens) < 3 or tokens[0] != 'cam':
            print(f"\nInvalid image name format: {extr.name}. Skipping.")
            continue
        camera_index = int(tokens[1].lstrip('0'))  # '01' -> 1
        frame_index = tokens[2]  # '001'

        camera_name = f"cam_{camera_index:02d}"
        frame_name = f"{camera_name}_{frame_index}"

        
        # Define paths for images and semantic features
        image_folder = os.path.join(images_root_folder, camera_name)
        semantic_feature_folder = os.path.join(semantic_features_root_folder, camera_name)

        # Check if directories exist
        if not os.path.isdir(image_folder):
            print(f"\nImage folder {image_folder} does not exist. Skipping camera {camera_name}.")
            continue
        if not os.path.isdir(semantic_feature_folder):
            print(f"\nSemantic feature folder {semantic_feature_folder} does not exist. Skipping camera {camera_name}.")
            continue

        # List and sort image files and semantic feature files
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        semantic_feature_files = sorted([f for f in os.listdir(semantic_feature_folder) if f.lower().endswith('.pt')])

        # Ensure that the number of images and semantic features match
        if len(image_files) != len(semantic_feature_files):
            print(f"\nMismatch in number of images and semantic features for camera {camera_name}. Skipping.")
            continue

        images = []
        image_paths = []
        image_names = []
        widths = []
        heights = []
        semantic_features = []
        semantic_feature_paths = []
        semantic_feature_names = []

        for img_file, feat_file in zip(image_files, semantic_feature_files):
            img_path = os.path.join(image_folder, img_file)
            feat_path = os.path.join(semantic_feature_folder, feat_file)

            # Load image
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"\nError loading image {img_path}: {e}. Skipping this image.")
                continue

            # Load semantic feature
            try:
                semantic_feature = torch.load(feat_path)
            except Exception as e:
                print(f"\nError loading semantic feature {feat_path}: {e}. Skipping this feature.")
                continue

            images.append(image)
            image_paths.append(img_path)
            image_names.append(os.path.splitext(img_file)[0])
            widths.append(width)
            heights.append(height)
            semantic_features.append(semantic_feature)
            semantic_feature_paths.append(feat_path)
            semantic_feature_names.append(os.path.splitext(feat_file)[0])

        if not images:
            print(f"\nNo valid images found for camera {camera_name}. Skipping.")
            continue

        # Calculate Field of View
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FoVy = focal2fov(focal_length_x, height)
            FoVx = focal2fov(focal_length_x, width)
        elif intr.model in ["PINHOLE", "OPENCV"]:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FoVy = focal2fov(focal_length_y, height)
            FoVx = focal2fov(focal_length_x, width)
        else:
            raise ValueError(f"Unsupported Colmap camera model: {intr.model}")

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FoVy=float(FoVy),
            FoVx=float(FoVx),
            images=images,
            image_paths=image_paths,
            image_names=image_names,
            widths=widths,
            heights=heights,
            semantic_features=semantic_features,
            semantic_feature_paths=semantic_feature_paths,
            semantic_feature_names=semantic_feature_names
        )
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


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

def readColmapSceneInfo(path, foundation_model, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    # Assuming images are organized per camera in subdirectories
    images_root_folder = os.path.join(path, "dynamic_rgb_images")
    semantic_features_root_folder = os.path.join(path, "dynamic_sam_embeddings")
    
    
    # Read FrameInfo objects
    camera_info_list = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_root_folder=images_root_folder, 
        semantic_features_root_folder=semantic_features_root_folder
    )
    
    # Sort frames by image name
    # frame_infos = sorted(frame_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    # Determine semantic feature dimension
    if camera_info_list and camera_info_list[0].frames[0].semantic_feature is not None:
        semantic_feature_dim = camera_info_list[0].frames[0].semantic_feature.shape[0]
    else:
        semantic_feature_dim = 0


    # Split frames into training and testing sets
    if eval:
        train_camera_infos = [c for idx, c in enumerate(camera_info_list) if idx % llffhold != 2]
        test_camera_infos = [c for idx, c in enumerate(camera_info_list) if idx % llffhold == 2]
    else:
        train_frame_infos = camera_info_list
        test_frame_infos = []



    # Get normalization parameters
    nerf_normalization = getNerfppNorm(train_frame_infos)

    # Load point cloud
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None



        
    # Create SceneInfo object
    scene_info = SceneInfo(
        point_cloud=pcd,
        cameras=train_frame_infos,  # Now lists of FrameInfo
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        semantic_feature_dim=semantic_feature_dim
    )


    # modified_scene_info = group_frames_by_camera_uid(scene_info=scene_info)
    
    return scene_info


def group_frames_by_camera_uid(scene_info):
    grouped_frames = defaultdict(list)
    for frame in scene_info.train_frames:
        uid = frame.camera_info.uid
        grouped_frames[uid].append(frame)
    # Convert the grouped_frames dict to a list of lists, sorted by uid
    grouped_train_frames = [grouped_frames[uid] for uid in sorted(grouped_frames.keys())]
    # Return a modified SceneInfo with grouped train_frames
    modified_scene_info = SceneInfo(
        point_cloud=scene_info.point_cloud,
        train_frames=grouped_train_frames,  # Now List[List[FrameInfo]]
        test_frames=scene_info.test_frames,
        nerf_normalization=scene_info.nerf_normalization,
        ply_path=scene_info.ply_path,
        semantic_feature_dim=scene_info.semantic_feature_dim
    )
    return modified_scene_info


def readStaticCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".png"):
    cam_infos = []

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

        filenames = contents.get("static_fn", [])

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


            # Construct the image file path
            image_path = os.path.join(path, "static", cam_name + extension)
            image_name = Path(cam_name).stem

            static_img_path = filenames[idx]
            image_path = static_img_path
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

            # Prepare frame information
            frames = []
            frame_info = FrameInfo(
                uid=idx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.width,
                height=image.height,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_name=None,
                time=None
            )
            frames.append(frame_info)

            # Process semantic features if provided (ensure paths are correct)

            # Create CameraInfo object
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                frames=frames
            )
            cam_infos.append(cam_info)

    return cam_infos

def readDynamicCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".jpg"):
    """
    Reads the transforms JSON file (e.g., train_meta.json) that now stores video information
    instead of image filenames. For each entry in the file, we extract the video path and 
    timestep, load that frame as a PIL image, and group frames by camera ID.
    """

    print(f"Reading cameras from {path} with {transformsfile}")
    # Open and load the transforms file
    meta_path = os.path.join(path, transformsfile)
    with open(meta_path, 'r') as json_file:
        contents = json.load(json_file)

    # Basic resolution info (assuming all frames share the same width and height)
    width = contents["width"]
    height = contents["height"]

    # For intrinsics, we assume either a single matrix or per-frame matrices.
    # Here we assume all frames share the same K.
    K = np.array(contents["k"][0]) if isinstance(contents["k"], list) else np.array(contents["k"])
    fx = K[0, 0]
    fy = K[1, 1]
    FoVx = 2 * np.arctan(width / (2 * fx))
    FoVy = 2 * np.arctan(height / (2 * fy))

    # Retrieve lists from the JSON (all lists should have the same length)
    w2c_list = contents.get("w2c", [])
    c2w_list = contents.get("c2w", [])
    k_list   = contents.get("k", [])
    cam_id_list = contents.get("cam_id", [])
    fn_list  = contents.get("fn", [])  # Now each element is a dict with "video_path" and "timestep"

    # Group frames by camera ID
    frames_by_cam = defaultdict(list)
    total_frames = len(cam_id_list)
    for idx in range(total_frames):
        cam_id = str(cam_id_list[idx])
        frame_data = {
            "w2c": w2c_list[idx],
            "c2w": c2w_list[idx],
            "k": np.array(k_list[idx]) if len(k_list) > 1 else K,
            "fn": fn_list[idx]  # Expecting dict: {"video_path": ..., "timestep": ...}
        }
        frames_by_cam[cam_id].append(frame_data)

    # Sort frames for each camera by their timestep
    for cam_id, frames in frames_by_cam.items():
        frames.sort(key=lambda f: f["fn"]["timestep"])

    # Now build CameraInfo objects
    cam_infos = []
    uid_global = 0  # Unique id counter for all frames
    for cam_id in sorted(frames_by_cam.keys(), key=lambda x: int(x)):
        print(f"Processing camera {cam_id}")
        camera_frames = []
        frame_datas = frames_by_cam[cam_id]
        for frame_data in frame_datas:
            video_path = frame_data["fn"]["video_path"]
            timestep = frame_data["fn"]["timestep"]

            # Open video capture, seek to the desired frame, and read it
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, timestep)
            ret, frame_img = cap.read()
            cap.release()
            if not ret:
                print(f"Warning: Could not read frame {timestep} from video {video_path}. Skipping this frame.")
                continue

            # Convert the frame from BGR (OpenCV format) to RGB and create a PIL Image
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_img)

            # Optionally handle background adjustments if needed.
            # (Here you could add similar logic as in your original code for handling alpha channels.)

            # Process extrinsics: use the provided c2w matrix, and adjust axes as before.
            c2w = np.array(frame_data["c2w"])
            # Adjust from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            # Create a FrameInfo object.
            # Here we use video_path as image_path and encode the timestep in image_name.
            frame_info = FrameInfo(
                uid=uid_global,
                image=image,
                image_path=video_path,
                image_name=f"{cam_id}_{timestep:04d}",
                width=image.width,
                height=image.height,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_name=None,
                time=timestep
            )
            uid_global += 1
            camera_frames.append(frame_info)

        # Skip cameras with no valid frames
        if len(camera_frames) == 0:
            continue

        # For the CameraInfo, you can choose to use the extrinsics from the first frame
        first_frame_data = frame_datas[0]
        c2w_first = np.array(first_frame_data["c2w"])
        c2w_first[:3, 1:3] *= -1
        w2c_first = np.linalg.inv(c2w_first)
        R_cam = np.transpose(w2c_first[:3, :3])
        T_cam = w2c_first[:3, 3]

        cam_info = CameraInfo(
            uid=int(cam_id),
            R=R_cam,
            T=T_cam,
            FoVy=FoVy,
            FoVx=FoVx,
            frames=camera_frames
        )
        cam_infos.append(cam_info)

    return cam_infos


def readCamerasFromTransforms_videoloading(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".jpg"):
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


def readCamerasFromTransforms_video(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".jpg"):
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

            # Load first frame of video
            video_path = contents["vid_path"][idx]
            video_name = Path(video_path).stem    
            video = cv2.VideoCapture(video_path)
            ret, frame = video.read()
            video.release()

            width = frame.shape[1]
            height = frame.shape[0]

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
                width=width,
                height=height,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_name=None,
                time=None
            )
            cam_infos.append(cam_info)

    return cam_infos


def readCamerasFromTransforms3(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # Retrieve the camera parameters from the JSON file
        width = contents["w"]
        height = contents["h"]

        # Extract the intrinsic matrix K
        K = np.array(contents["k"])[0][0]

        # Extract the focal lengths
        fx = K[0, 0]
        fy = K[1, 1]

        # Compute horizontal and vertical FoV in degrees
        FoVx = 2 * np.arctan(width / (2 * fx)) # * 180 / np.pi
        FoVy = 2 * np.arctan(height / (2 * fy))# * 180 / np.pi

        # Get camera world-to-camera matrices
        w2c_matrices = contents.get("w2c", [])[0]

        for idx, w2c_matrix in enumerate(w2c_matrices):
            cam_name = f"cam_{idx:03d}"  # Camera name with leading zeros

            # Convert w2c_matrix to numpy array
            w2c = np.array(w2c_matrix)

            # Define the rotation matrix (180 degrees around X-axis)
            rotation_matrix = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ])

            # Adjust the w2c matrix
            w2c = rotation_matrix @ w2c # original
            # w2c = w2c @ rotation_matrix

            

            # Extract rotation and translation
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            # T = - R @ w2c[:3, 3]

            # Construct the image file path
            image_path = os.path.join(path, "images", cam_name + extension)
            image_name = Path(cam_name).stem

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

            # Prepare frame information
            frames = []
            frame_info = FrameInfo(
                uid=idx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.width,
                height=image.height,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_name=None,
                time=None
            )
            frames.append(frame_info)

            # Process semantic features if provided (ensure paths are correct)

            # Create CameraInfo object
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                frames=frames
            )
            cam_infos.append(cam_info)

    return cam_infos



def readCamerasFromTransforms_1(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # Retrieve the camera parameters from the JSON file
        width = contents.get("w")
        height = contents.get("h")


        K = np.array(contents["k"])[0][0]

        # Extract the focal lengths
        fx = K[0, 0]  # Horizontal focal length
        fy = K[1, 1]  # Vertical focal length

        # Assuming you have the image dimensions (width and height) available
        width = contents["w"]  # Image width
        height = contents["h"]  # Image height



        # Compute horizontal FoV (FoVx)
        FoVx = 2 * np.arctan(width / (2 * fx))#  * 180 / np.pi  # Convert from radians to degrees

        FoVy = 2 * np.arctan(height / (2 * fy))#  * 180 / np.pi  # Convert from radians to degrees

        # FoVx = contents.get("camera_angle_x", None)
        # FoVy = None  # Placeholder for vertical FoV if available

        # Get camera world-to-camera matrices
        w2c_matrices = contents.get("w2c", [])[0] if contents.get("w2c") else []

        for idx, w2c_matrix in enumerate(w2c_matrices):
            cam_name = f"cam_{idx:03d}"  # Camera name with leading zeros

            # NeRF 'w2c' is already the world-to-camera transform
            w2c = np.array(w2c_matrix)
            R = np.transpose(w2c[:3, :3])  # Transpose for correct orientation
            T = w2c[:3, 3]

            # Construct the image file path
            image_path = os.path.join(path, "images", cam_name + extension)
            image_name = Path(cam_name).stem

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

            # Prepare to collect frame information
            frames = []
            frame_info = FrameInfo(
                uid=idx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.width,
                height=image.height,
                semantic_feature=None,
                semantic_feature_path=None,
                semantic_feature_name=None,
                time=None  # Adjust this if you have time data
            )
            frames.append(frame_info)

            # Check if semantic features are provided
            if semantic_feature_root_folder:
                camera_dir = os.path.join(semantic_feature_root_folder, image_name)
                semantic_feature_folder = camera_dir
                if not os.path.isdir(semantic_feature_folder):
                    print(f"Semantic feature folder {semantic_feature_folder} does not exist. Skipping camera {image_name}.")
                    continue

                # List and sort semantic feature files
                semantic_feature_files = sorted([f for f in os.listdir(semantic_feature_folder) if f.lower().endswith('.pt')])

                for feat_file in semantic_feature_files:
                    img_file = f"{image_name.split('_')[0]}_{feat_file.split('_')[1]}.png"
                    img_path = os.path.join(path, "dynamic_rgb_images", image_name.split('_')[0], img_file)
                    feat_path = os.path.join(semantic_feature_folder, feat_file)

                    # Load semantic feature
                    try:
                        semantic_feature = torch.load(feat_path)
                    except Exception as e:
                        print(f"Error loading semantic feature {feat_path}: {e}. Skipping this feature.")
                        continue

                    # Add semantic feature data to the frame info
                    frame_info.semantic_feature = semantic_feature
                    frame_info.semantic_feature_path = feat_path
                    frame_info.semantic_feature_name = os.path.splitext(feat_file)[0]

            # Create CameraInfo object
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                frames=frames  # Attach the frames list
            )
            cam_infos.append(cam_info)

    return cam_infos



def readCamerasFromTransforms_original(path, transformsfile, white_background, semantic_feature_root_folder, extension=".jpg"): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        FoVx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping this image.")
                continue

            # Handle alpha channel and background
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray((arr*255.0).astype(np.uint8), "RGB")

            foVy = focal2fov(fov2focal(FoVx, image.size[0]), image.size[1])
            FoVy = float(foVy)
            FoVx = float(FoVx)

            # Assuming semantic features are stored per camera
            camera_dir = os.path.join(semantic_feature_root_folder, image_name)
            semantic_feature_folder = camera_dir  # Assuming each camera has its own folder
            if not os.path.isdir(semantic_feature_folder):
                print(f"Semantic feature folder {semantic_feature_folder} does not exist. Skipping camera {image_name}.")
                continue

            # List and sort semantic feature files
            semantic_feature_files = sorted([f for f in os.listdir(semantic_feature_folder) if f.lower().endswith('.pt')])

            images = []
            image_paths = []
            image_names = []
            widths = []
            heights = []
            semantic_features = []
            semantic_feature_paths = []
            semantic_feature_names = []

            for feat_file in semantic_feature_files:
                img_file = f"{image_name.split('_')[0]}_{feat_file.split('_')[1]}.jpg"  # Assuming naming convention
                img_path = os.path.join(path, "dynamic_rgb_images", image_name.split('_')[0], img_file)
                feat_path = os.path.join(semantic_feature_folder, feat_file)

                # Load image
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}. Skipping this image.")
                    continue

                # Load semantic feature
                try:
                    semantic_feature = torch.load(feat_path)
                except Exception as e:
                    print(f"Error loading semantic feature {feat_path}: {e}. Skipping this feature.")
                    continue

                images.append(img)
                image_paths.append(img_path)
                image_names.append(os.path.splitext(img_file)[0])
                widths.append(img.width)
                heights.append(img.height)
                semantic_features.append(semantic_feature)
                semantic_feature_paths.append(feat_path)
                semantic_feature_names.append(os.path.splitext(feat_file)[0])

            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                images=images,
                image_paths=image_paths,
                image_names=image_names,
                widths=widths,
                heights=heights,
                semantic_features=semantic_features,
                semantic_feature_paths=semantic_feature_paths,
                semantic_feature_names=semantic_feature_names
            )
            cam_infos.append(cam_info)

    return cam_infos




def read_cameras_from_blender(path, blender_json_file, white_background=False, semantic_feature_root_folder=None, use_semantic_features=False, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, blender_json_file)) as json_file:
        contents = json.load(json_file)
        
        for idx, camera in enumerate(contents):
            cam_name = camera["filename"]

            # Camera extrinsics and intrinsics
            c2w = np.array(camera["camera_extrinsics"])
            intrinsics = np.array(camera["camera_intrinsics"])
            
            # Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # Get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Load image
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping this image.")
                continue

            # Handle alpha channel and background
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray((arr * 255.0).astype(np.uint8), "RGB")

            FoVx = 2 * np.arctan(image.size[0] / (2 * intrinsics[0, 0])) * 180 / np.pi
            FoVy = 2 * np.arctan(image.size[1] / (2 * intrinsics[1, 1])) * 180 / np.pi

            semantic_features = []
            semantic_feature_paths = []
            semantic_feature_names = []

            if use_semantic_features:
                # Assuming semantic features are stored per camera
                camera_dir = os.path.join(semantic_feature_root_folder, image_name)
                semantic_feature_folder = camera_dir  # Assuming each camera has its own folder
                if not os.path.isdir(semantic_feature_folder):
                    print(f"Semantic feature folder {semantic_feature_folder} does not exist. Skipping camera {image_name}.")
                    continue

                # List and sort semantic feature files
                semantic_feature_files = sorted([f for f in os.listdir(semantic_feature_folder) if f.lower().endswith('.pt')])

                for feat_file in semantic_feature_files:
                    feat_path = os.path.join(semantic_feature_folder, feat_file)

                    # Load semantic feature
                    try:
                        semantic_feature = torch.load(feat_path)
                    except Exception as e:
                        print(f"Error loading semantic feature {feat_path}: {e}. Skipping this feature.")
                        continue

                    semantic_features.append(semantic_feature)
                    semantic_feature_paths.append(feat_path)
                    semantic_feature_names.append(os.path.splitext(feat_file)[0])

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
                semantic_features=semantic_features,
                semantic_feature_paths=semantic_feature_paths,
                semantic_feature_names=semantic_feature_names
            )
            cam_infos.append(cam_info)

    return cam_infos



def read_blender_scene(path, foundation_model, white_background, eval, extension=".png"):
    if foundation_model == 'sam':
        semantic_feature_dir = "sam_embeddings"
    elif foundation_model == 'lseg':
        semantic_feature_dir = "rgb_feature_langseg"
    else:
        semantic_feature_dir = ""

    use_semantic_features = bool(semantic_feature_dir)

    print("Reading Transforms")
    cam_infos = read_cameras_from_blender(path, "blender.json", white_background, semantic_feature_root_folder=os.path.join(path, semantic_feature_dir), use_semantic_features=use_semantic_features)

    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 50_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    semantic_feature_dim = cam_infos[0].semantic_features[0].shape[0] if use_semantic_features and cam_infos[0].semantic_features else 0
    scene_info = SceneInfo(point_cloud=pcd,
                           cameras=cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info


def readBlenderSyntheticInfo(path, white_background, foundation_model=None):

    if foundation_model == 'sam':
        semantic_feature_dir = "sam_embeddings"
    elif foundation_model == 'lseg':
        semantic_feature_dir = "rgb_feature_langseg"
    else:
        semantic_feature_dir = ""

    use_semantic_features = bool(semantic_feature_dir)

    print("Reading Transforms")
    # cam_infos = readCamerasFromTransforms(path, "train_meta.json", white_background, semantic_feature_root_folder=None)
    cam_infos = readCamerasFromTransforms_video(path, "train_meta.json", white_background, semantic_feature_root_folder=None)

    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 6 - 3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    semantic_feature_dim = cam_infos[0].semantic_features[0].shape[0] if use_semantic_features and cam_infos[0].semantic_features else 0
    scene_info = SceneInfo(point_cloud=pcd,
                        cameras=cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path,
                        semantic_feature_dim=semantic_feature_dim) 
    return scene_info


def readStaticBlenderSyntheticInfo1(path, white_background, foundation_model=None):

    if foundation_model == 'sam':
        semantic_feature_dir = "sam_embeddings"
    elif foundation_model == 'lseg':
        semantic_feature_dir = "rgb_feature_langseg"
    else:
        semantic_feature_dir = ""

    use_semantic_features = bool(semantic_feature_dir)

    print("Reading Transforms")
    # cam_infos = readStaticCamerasFromTransforms(path, "train_static_meta.json", white_background, semantic_feature_root_folder=None)
    cam_infos = readCamerasFromTransforms(path, "train_meta.json", white_background, semantic_feature_root_folder=None)
    # cam_infos = readDynamicCamerasFromTransforms(path, "train_dynamic_meta.json", white_background, semantic_feature_root_folder=None)
    
    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 60 - 30
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    semantic_feature_dim = cam_infos[0].semantic_features[0].shape[0] if use_semantic_features and cam_infos[0].semantic_features else 0
    scene_info = SceneInfo(point_cloud=pcd,
                        cameras=cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path,
                        semantic_feature_dim=semantic_feature_dim) 
    return scene_info


def readNerfSyntheticInfo(path, foundation_model, white_background, eval, extension=".png"): 
    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3D.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0] 
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}