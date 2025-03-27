# scene/__init__.py
import numpy as np
import torch
import torch.nn as nn
import os

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import Camera
from scene.blender_dataset_reader import readSceneInfoBlender
from utils.camera_utils import cameraObjectsNoImage_from_cameraInfos


class Scene:
    def __init__(self, config, scene_path=None):
        """Initialize an empty scene or from a dataset."""
        self.gaussians = None
        self.train_cameras = {}
        self.test_cameras = {}
        self.dataset_path = scene_path

        print("Read Scene Info")
        self.scene_info = readSceneInfoBlender(self.dataset_path, white_background=False)
        
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        
        print("Create Training Camera Objects")
        self.train_camera_objects = cameraObjectsNoImage_from_cameraInfos(camera_infos=self.scene_info.cameras, args=config)
        self.train_cameras = self.scene_info.cameras


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


    def load_gaussians_from_checkpoint(self, checkpoint_path, gaussians, opt):
        """
        Load Gaussians from a checkpoint and assign them to the scene.
        """
        if self.gaussians is None:
            self.gaussians = gaussians
        model_params = torch.load(checkpoint_path)
        self.gaussians.restore(model_params, opt)
        print("Gaussians loaded from checkpoint.")

    def load_gaussians_from_checkpoint_for_msgnode(self, checkpoint_path, gaussians):
        if self.gaussians is None:
            self.gaussians = gaussians
        model_params = torch.load(checkpoint_path)
        self.gaussians.restore_msgnode(model_params)
        print("Gaussians loaded from checkpoint.")

    def getTrainCameras(self):
        return self.train_cameras


    def getTrainCameraObjects(self):
        return self.train_camera_objects
    
    def getTestCameraObjects(self):
        return self.test_camera_objects
    