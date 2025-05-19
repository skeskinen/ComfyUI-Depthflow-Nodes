import torch
from DepthFlow.Scene import DepthScene
from DepthFlow.Animation import Animation, Target

import numpy as np
from collections import deque
from comfy.utils import ProgressBar
import gc
import subprocess
import sys
import re
import os
from PIL import Image
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

class CustomDepthflowScene(DepthScene):

    def __init__(
        self,
        state=None,
        effects=None,
        progress_callback=None,
        **kwargs,
    ):
        DepthScene.__init__(self, **kwargs)
        self.frames = deque()
        self.progress_callback = progress_callback
        # Initialize images and depth_maps
        self.images = None
        self.depth_maps = None

    def _load_inputs(self, echo: bool=True) -> None:
        """Load inputs: single or batch exporting"""
        # Batch exporting implementation
        image = self._get_batch_input(self.config.image)
        depth = self._get_batch_input(self.config.depth)

        if (image is None):
            raise ShaderBatchStop()

        # Convert numpy arrays to PIL Images properly
        # The error occurs because PIL.Image.fromarray() can't handle the data type directly
        # Make sure arrays are in the right format (uint8 for RGB images)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if depth is not None and depth.dtype != np.uint8:
            depth = (depth * 255).astype(np.uint8)

        image_pil = Image.fromarray(image)
        if depth is not None:
            depth_pil = Image.fromarray(depth)
        else:
            depth_pil = self.config.estimator.estimate(image)

        # Match rendering resolution to image
        self.resolution = (image_pil.width, image_pil.height)
        self.aspect_ratio = (image_pil.width / image_pil.height)
        self.image.from_image(image_pil)
        self.depth.from_image(depth_pil)

    def next(self, dt):
        DepthScene.next(self, dt)
        width, height = self.resolution
        array = np.frombuffer(self._final.texture.fbo.read(), dtype=np.uint8).reshape(
            (height, width, 3)
        )

        array = np.flip(array, axis=0).copy()

        # To Tensor
        tensor = torch.from_numpy(array)

        del array

        # Accumulate the frame
        self.frames.append(tensor)

        if self.progress_callback:
            self.progress_callback()

        return self

    def get_accumulated_frames(self):
        # Convert the deque of frames to a tensor
        return torch.stack(list(self.frames))

    def clear_frames(self):
        self.frames.clear()
        gc.collect()

class Depthflow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image
                "animation_speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "step": 0.01},
                ),
                "input_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "step": 1.0}),
                "output_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "step": 1.0}),
                "num_frames": ("INT", {"default": 30, "min": 1, "step": 1}),
                "quality": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "ssaa": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "invert": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "animation_intensity": (
                    "FLOAT",
                    {"default": 1.5, "min": -5.0, "max": 5.0, "step": 0.1},
                ),
                "animation_mode": (["zoom", "dolly", "circle", "horizontal", "vertical"], {"default": "zoom"}),
                "animation_smooth": (
                    "BOOLEAN",
                    {"default": False, "label": "Smooth"},
                ),
                "animation_loop": (
                    "BOOLEAN",
                    {"default": False, "label": "Loop"},
                ),
                "inpaint": (
                    "BOOLEAN",
                    {"default": False, "label": "Inpaint"},
                ),
                "inpaint_black": (
                    "BOOLEAN",
                    {"default": False, "label": "Inpaint Black"},
                ),
                "tiling_mode": (["mirror", "repeat", "none"], {"default": "mirror"}),
            },
            "optional": {
                "depth_map": ("IMAGE",),  # Depthmap input
                "motion": ("DEPTHFLOW_MOTION",),  # Motion object
                "effects": ("DEPTHFLOW_EFFECTS",),  # DepthState object
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )  # Output is a batch of images (torch.Tensor with shape [B,H,W,C])
    FUNCTION = "apply_depthflow"
    CATEGORY = "🌊 Depthflow"
    DESCRIPTION = """
    Depthflow Node:
    This node applies a motion animation (Zoom, Dolly, Circle, Horizontal, Vertical) to an image
    using a depthmap and outputs an image batch as a tensor.
    - image: The input image.
    - depth_map: Depthmap corresponding to the image.
    - options: DepthState object.
    - motion: Depthflow motion object.
    - input_fps: Frames per second for the input video.
    - output_fps: Frames per second for the output video.
    - num_frames: Number of frames for the output video.
    - quality: Quality of the output video.
    - ssaa: Super sampling anti-aliasing samples.
    - invert: Invert the depthmap.
    - tiling_mode: Tiling mode for the image.
    """

    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

    def apply_depthflow(
        self,
        image,
        animation_speed,
        input_fps,
        output_fps,
        num_frames,
        quality,
        ssaa,
        invert,
        tiling_mode,
        animation_intensity,
        animation_mode,
        animation_smooth,
        animation_loop,
        inpaint,
        inpaint_black,
        depth_map=None,
        motion=None,
        effects=None,
    ):
        # Create the scene
        state = {"invert": invert, "tiling_mode": tiling_mode}
        scene = CustomDepthflowScene(
            # state=state,
            # effects=effects,
            # progress_callback=self.update_progress,
            # num_frames=num_frames,
            # input_fps=input_fps,
            # output_fps=output_fps,
            # animation_speed=animation_speed,
            backend="headless",
        )


        # Convert image and depthmap to numpy arrays
        if image.is_cuda:
            image = image.cpu().numpy()
        else:
            image = image.numpy()
        if depth_map is not None and depth_map.is_cuda:
            depth_map = depth_map.cpu().numpy()
        elif depth_map is not None:
            depth_map = depth_map.numpy()

        if image.ndim != 4:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if depth_map is not None and depth_map.ndim != 4:
            raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")

        if image.shape[0] != 1:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if depth_map is not None and depth_map.shape[0] != 1:
            raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")

        # Get width and height of images
        height, width = image.shape[1], image.shape[2]

        # Input the image and depthmap into the scene
        scene.input(image[0], depth=depth_map[0] if depth_map is not None else None)

        # Alternatively, if your scene has a helper method:
        if inpaint:
            scene.inpaint(
                enable=True,
                black=inpaint_black,
                limit=0.8
            )

        if animation_mode == "zoom":
            scene.zoom(
                intensity=animation_intensity,
                smooth=animation_smooth,
                loop=animation_loop
            )
        elif animation_mode == "dolly":
            scene.dolly(
                intensity=animation_intensity,
                smooth=animation_smooth,
                loop=animation_loop
            )
        elif animation_mode == "circle":
            scene.circle(
                intensity=animation_intensity,
                smooth=animation_smooth,
                loop=animation_loop
            )
        elif animation_mode == "horizontal":
            scene.horizontal(
                intensity=animation_intensity,
                smooth=animation_smooth,
                loop=animation_loop,
                # phase=0.5
            )

        # Calculate the duration based on fps and num_frames
        if num_frames <= 0:
            raise ValueError("FPS and number of frames must be greater than 0")
        duration = float(num_frames) / input_fps
        total_frames = duration * output_fps

        self.start_progress(total_frames, desc="Depthflow Rendering")

        # Render the output video
        scene.main(
            render=False,
            output=None,
            fps=output_fps,
            time=duration,
            speed=1.0,
            start=0.0,
            quality=quality,
            ssaa=ssaa,
            width=width,
            height=height,
            freewheel=True,
        )

        video = scene.get_accumulated_frames()
        scene.clear_frames()
        self.end_progress()

        # Create mask from the green channel
        # Extract green channel (index 1), check if values are at maximum (255)
        mask = (video[:, :, :, 1] == 255).float()

        # Normalize the video frames to [0, 1]
        video = video.float() / 255.0

        return (video, mask)
