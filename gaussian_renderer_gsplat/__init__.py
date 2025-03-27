#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from torch.nn import functional as F
from gsplat import rasterization
from scene.gaussian_model import GaussianModel

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.width / 2.0],
            [0, focal_length_y, viewpoint_camera.height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.width),
        height=int(viewpoint_camera.height),
        packed=False,
        sh_degree=sh_degree,
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_batch(viewpoint_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if isinstance(viewpoint_cameras, list):
        viewpoint_camera = viewpoint_cameras[0]
        batch_size = len(viewpoint_cameras)
    else:
        viewpoint_camera = viewpoint_cameras
        batch_size = 1

    # viewpoint_camera.width = 500
    # viewpoint_camera.height = 500

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.width / 2.0],
            [0, focal_length_y, viewpoint_camera.height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    Ks = []
    for cam in viewpoint_cameras:
        Ks.append(K)
    Ks = torch.stack(Ks)

    # batch_size = viewpoint_cameras.shape[0]
    # Ks = K.repeat(batch_size, 1, 1)


    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree


    #viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    viewmats = []
    for cam in viewpoint_cameras:
        viewmats.append(cam.world_view_transform.transpose(0, 1))
    viewmats = torch.stack(viewmats)


    bg_colors = bg_color[None].repeat(batch_size, 1)
    
    width = int(viewpoint_camera.width)
    height = int(viewpoint_camera.height)
    
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmats,  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        backgrounds=bg_colors,
        width=width,
        height=height,
        packed=False,
        sh_degree=sh_degree,
    ) # [Cameras, H, W, 3]


    rendered_images = render_colors.permute(0, 3, 1, 2) # [Cameras, channels, H, W]
    radii = info["radii"] # [Cameras, N]
    means2d = info["means2d"] # [Cameras, N, 2]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_images,
            "viewspace_points": means2d,
            "visibility_filter" : radii > 0,
            "radii": radii}
