# scene/__init__.py

import os
import random
import pickle
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, read_blender_scene, readBlenderSyntheticInfo
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraObjects_from_cameraInfos
import torch

class Scene:

    def __init__(self, args: ModelParams, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """
        Initialize the Scene without loading or creating Gaussians.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = None  # Initialize gaussians as None
        self.train_cameras = {}
        self.test_cameras = {}
        self.scene_info = None
        self.cameras_extent = None

        # Load scene_info
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.foundation_model, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "blender.json")):
            print("Found blender.json, assuming custom Blender data set!")
            self.scene_info = read_blender_scene(args.source_path, "blender.json", white_background=args.white_background, eval=args.eval)
        elif os.path.exists(os.path.join(args.source_path, "train_meta.json")):
            print("Found train_meta.json, assuming custom SyntheticBlender dataset!")
            self.scene_info = readBlenderSyntheticInfo(args.source_path, white_background=args.white_background)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            self.scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.foundation_model, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        if shuffle:
            random.shuffle(self.scene_info.cameras)  # Multi-res consistent random shuffling

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_camera_objects = cameraObjects_from_cameraInfos(self.scene_info.cameras, args)
            self.train_cameras = self.scene_info.cameras

    def load_gaussians_from_checkpoint(self, checkpoint_path, gaussians, opt):
        """
        Load Gaussians from a checkpoint and assign them to the scene.
        """
        if self.gaussians is None:
            self.gaussians = gaussians
        model_params, _ = torch.load(checkpoint_path)
        self.gaussians.restore(model_params, opt)
        print("Gaussians loaded from checkpoint.")

    def create_gaussians_manually(self, gaussians):
        """
        Manually assign Gaussians to the scene.
        """
        self.gaussians = gaussians
        print("Gaussians created manually.")

    def initialize_gaussians_from_scene_info(self, gaussians, args):
        """
        Initialize Gaussians from scene information.
        """
        if self.gaussians is None:
            self.gaussians = gaussians
        self.gaussians.create_from_pcd(
            self.scene_info.point_cloud,
            self.cameras_extent,
            self.scene_info.semantic_feature_dim,
            args.speedup
        )
        print("Gaussians initialized from scene information.")

    # The rest of your Scene methods remain unchanged

    def save_scene_attributes(self, filepath):
        """
        Saves only the necessary Scene attributes using pickle.
        """
        # Create a dictionary of essential attributes
        scene_data = {
            "model_path": self.model_path,
            "loaded_iter": self.loaded_iter,
            "gaussians": self.gaussians,
            "train_cameras": self.train_cameras,
            "test_cameras": self.test_cameras,
            "scene_info": self.scene_info,
            "cameras_extent": self.cameras_extent
        }

        # Save this dictionary to a file using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(scene_data, f)
        print(f"Scene attributes saved to {filepath}")

    @staticmethod
    def load_scene_attributes(filepath):
        """
        Loads only the necessary Scene attributes from a pickle file.
        """
        # Load the dictionary of saved attributes
        with open(filepath, 'rb') as f:
            scene_data = pickle.load(f)
        print(f"Scene attributes loaded from {filepath}")
        return scene_data

    def restore_from_attributes(self, scene_data):
        """
        Restores the Scene object from a dictionary of attributes.
        """
        self.model_path = scene_data["model_path"]
        self.loaded_iter = scene_data["loaded_iter"]
        self.gaussians = scene_data["gaussians"]
        self.train_cameras = scene_data["train_cameras"]
        self.test_cameras = scene_data["test_cameras"]
        self.scene_info = scene_data["scene_info"]
        self.cameras_extent = scene_data["cameras_extent"]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTrainCameraObjects(self):
        return self.train_camera_objects

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
