# utils/camera_utils.py

# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera, Camera_noImage
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scene.dataset_readers import FrameInfo
import torch

WARNED = False

def loadCam_original(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    
    gt_semantic_feature = cam_info.semantic_feature
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    
    # image size will the same as feature map size
    elif args.resolution == 0:
        resolution = gt_semantic_feature.shape[2], gt_semantic_feature.shape[1]    
    # customize resolution
    elif args.resolution == -2:
        resolution = 480, 320 #800, 450

    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]


    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, semantic_feature = gt_semantic_feature,
                  data_device=args.data_device) 

def loadCam(args, id, frame: FrameInfo, resolution_scale):
    orig_w, orig_h = frame.image.size
    gt_semantic_feature = frame.semantic_feature

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
    elif args.resolution == 0:
        resolution = gt_semantic_feature.shape[2], gt_semantic_feature.shape[1]
    elif args.resolution == -2:
        resolution = 480, 320
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered large images (>1.6K pixels width), rescaling to 1.6K.\n"
                          "If this is not desired, specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(frame.image, resolution)
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=frame.uid,
        R=frame.camera_info.R,
        T=frame.camera_info.T,
        FoVx=frame.camera_info.FovX,
        FoVy=frame.camera_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=frame.image_name,
        uid=id,
        semantic_feature=gt_semantic_feature,
        data_device=args.data_device
    )

def createCamObject_noImage(args, camera_infos):
    return Camera_noImage(
        colmap_id=camera_infos.uid,
        R=camera_infos.R,
        T=camera_infos.T,
        FoVx=camera_infos.FoVx,
        FoVy=camera_infos.FoVy,
        width=camera_infos.width,
        height=camera_infos.height,
        uid=camera_infos.uid,
        data_device=args.data_device
    )

def createCamObject(args, camera_infos):

    loaded_mask = None
    gt_image = torch.from_numpy(np.array(camera_infos.frames[0].image)) / 255.0
    gt_semantic_feature = camera_infos.frames[0].semantic_feature
    image_name = camera_infos.frames[0].image_name

    return Camera(
    colmap_id=camera_infos.uid,
    R=camera_infos.R,
    T=camera_infos.T,
    FoVx=camera_infos.FoVx,
    FoVy=camera_infos.FoVy,
    image=gt_image,
    gt_alpha_mask=loaded_mask,
    image_name=image_name,
    uid=id,
    semantic_feature=gt_semantic_feature,
    data_device=args.data_device)



def createCamObject_video(args, camera_info):
    gt_image = torch.from_numpy(np.array(camera_info.image)).float().cuda() / 255.0
    gt_semantic_feature = camera_info.semantic_feature
    loaded_mask = torch.ones_like(gt_image).cuda()
    gt_video = camera_info.video

    return Camera(
        colmap_id=camera_info.uid,
        R=camera_info.R,
        T=camera_info.T,
        FoVx=camera_info.FoVx,
        FoVy=camera_info.FoVy,
        image=gt_image,
        video=gt_video,
        gt_alpha_mask=loaded_mask,
        image_name=camera_info.image_name,
        uid=camera_info.uid,
        semantic_feature=gt_semantic_feature,
        data_device=args.data_device
    )


def cameraList_from_camInfos_original(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def cameraList_from_frameInfos(frame_infos, resolution_scale, args):
    camera_list = []
    for id, frame_info in enumerate(frame_infos):
        camera_list.append(loadCam(args, id, frame_info, resolution_scale))
    return camera_list

def cameraObjectsNoImage_from_cameraInfos(camera_infos, args):
    camera_objects = []
    for id, camera_info in enumerate(camera_infos):
        camera_objects.append(createCamObject_noImage(args, camera_info))
    return camera_objects

def cameraObjects_from_cameraInfos(camera_infos, args):
    camera_objects = []
    for id, camera_info in enumerate(camera_infos):
        camera_objects.append(createCamObject(args, camera_info))
    return camera_objects

def camera_to_JSON_original(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def camera_to_JSON(id, frame: FrameInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = frame.camera_info.R.transpose()
    Rt[:3, 3] = frame.camera_info.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = rot.tolist()

    camera_entry = {
        'id': id,
        'img_name': frame.image_name,
        'width': frame.width,
        'height': frame.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(frame.camera_info.FovY, frame.height),
        'fx': fov2focal(frame.camera_info.FovX, frame.width)
    }
    return camera_entry