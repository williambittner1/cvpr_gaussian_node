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
    
    return scene_info




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


def read_cameras_from_meta_json(path, transformsfile, white_background, semantic_feature_root_folder=None, extension=".jpg"):
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




def readSceneInfoBlender(path, white_background, foundation_model=None):

    if foundation_model == 'sam':
        semantic_feature_dir = "sam_embeddings"
    elif foundation_model == 'lseg':
        semantic_feature_dir = "rgb_feature_langseg"
    else:
        semantic_feature_dir = ""

    use_semantic_features = bool(semantic_feature_dir)

    print("Reading cam_infos from train_meta.json")
    cam_infos = read_cameras_from_meta_json(path, "train_meta.json", white_background, semantic_feature_root_folder=None)

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

    semantic_feature_dim = cam_infos[0].semantic_features[0].shape[0] if use_semantic_features and cam_infos[0].semantic_features else 0

    scene_info = SceneInfo(point_cloud=pcd,
                        cameras=cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path,
                        semantic_feature_dim=semantic_feature_dim) 
    return scene_info

