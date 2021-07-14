from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointsRasterizationSettings, PointsRenderer, PointsRasterizer,
    look_at_view_transform,
    HardPhongShader,
    AlphaCompositor,
)

def points_renderer(device):
    # Initialize a perspective camera.
    cameras = PerspectiveCameras(
        device=device,
    )

    points_raster_settings = PointsRasterizationSettings(
        image_size=224,
        radius = 0.01,
        points_per_pixel = 140,
    )

    return PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=cameras,
            raster_settings=points_raster_settings
        ),
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    )

def mesh_renderer(device):
    # Initialize a perspective camera.
    cameras = PerspectiveCameras(
        device=device,
    )

    raster_settings = RasterizationSettings(
        image_size=224,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False
    )

    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras)
    )
