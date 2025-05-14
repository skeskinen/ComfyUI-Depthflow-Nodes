import torch
from DepthFlow.Scene import DepthScene
from DepthFlow.Animation import Animation
from Broken.Loaders import LoadableImage, LoadImage

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
        num_frames=30,
        input_fps=30.0,
        output_fps=30.0,
        animation_speed=1.0,
        **kwargs,
    ):
        DepthScene.__init__(self, **kwargs)
        self.frames = deque()
        self.progress_callback = progress_callback
        # Override state with keywords in state
        self.override_state = state
        self.time = 0.00001
        # Initialize images and depth_maps
        self.images = None
        self.depth_maps = None
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.animation_speed = animation_speed
        self.num_frames = num_frames
        self.video_time = 0.0
        self.frame_index = 0

    # # TODO: This is a temporary fix to while build gets fixed
    # def build(self):
    #     self.image = ShaderTexture(scene=self, name="image").repeat(False)
    #     self.depth = ShaderTexture(scene=self, name="depth").repeat(False)
    #     self.normal = ShaderTexture(scene=self, name="normal")
    #     self.shader.fragment = self.DEPTH_SHADER
    #     self.ssaa = 1.2

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
        if depth.dtype != np.uint8:
            depth = (depth * 255).astype(np.uint8)

        image_pil = Image.fromarray(image)
        depth_pil = Image.fromarray(depth)

        # Match rendering resolution to image
        self.resolution = (image_pil.width, image_pil.height)
        self.aspect_ratio = (image_pil.width / image_pil.height)
        self.image.from_image(image_pil)
        self.depth.from_image(depth_pil)

    def setup(self):
        DepthScene.setup(self)
        self.time += 0.00001  # prevent division by zero error

    # def update(self):
    #     self.log_info("update called")
    #     frame_duration = 1.0 / self.input_fps

    #     while self.time > self.video_time:
    #         self.video_time += frame_duration
    #         self.frame_index += 1

    #     # Set the current image and depth map based on self.frame
    #     if self.images is not None and self.depth_maps is not None:
    #         frame_index = min(self.frame_index, len(self.images) - 1)
    #         current_image = self.images[frame_index]
    #         current_depth = self.depth_maps[frame_index]

    #         # Convert to appropriate format if necessary
    #         image = self.upscaler.upscale(LoadImage(current_image))
    #         depth = LoadImage(current_depth)

    #         # Set the current image and depth map
    #         self.image.from_image(image)
    #         self.depth.from_image(depth)

    #     DepthScene.update(self)

    #     if self.override_state:
    #         for key, value in self.override_state.items():
    #             if hasattr(self.state, key):
    #                 setattr(self.state, key, value)

    #         if "tiling_mode" in self.override_state:
    #             if self.override_state["tiling_mode"] == "repeat":
    #                 self.image.repeat(True)
    #                 self.depth.repeat(True)
    #                 self.state.mirror = False
    #             elif self.override_state["tiling_mode"] == "mirror":
    #                 self.image.repeat(False)
    #                 self.depth.repeat(False)
    #                 self.state.mirror = True
    #             else:
    #                 self.image.repeat(False)
    #                 self.depth.repeat(False)
    #                 self.state.mirror = False

    @property
    def tau(self) -> float:
        return super().tau * self.animation_speed

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


class MyDepthflowScene(DepthScene):

    def _load_inputs(self, echo: bool=True) -> None:
        """Load inputs: single or batch exporting"""
        # Batch exporting implementation
        image = self._get_batch_input(self.config.image)
        depth = self._get_batch_input(self.config.depth)

        if (image is None):
            raise ShaderBatchStop()

        self.log_warn("hellurei")

        self.log_info(f'image.shape: {image.shape}', echo=echo)
        self.log_info(f'depth.shape: {depth.shape}', echo=echo)

        # Convert numpy arrays to PIL Images properly
        # The error occurs because PIL.Image.fromarray() can't handle the data type directly
        # Make sure arrays are in the right format (uint8 for RGB images)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if depth.dtype != np.uint8:
            depth = (depth * 255).astype(np.uint8)

        image_pil = Image.fromarray(image)
        depth_pil = Image.fromarray(depth)

        # Match rendering resolution to image
        self.resolution = (image_pil.width, image_pil.height)
        self.aspect_ratio = (image_pil.width / image_pil.height)
        self.image.from_image(image_pil)
        self.depth.from_image(depth_pil)


class Depthflow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image
                "depth_map": ("IMAGE",),  # Depthmap input
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
                "tiling_mode": (["mirror", "repeat", "none"], {"default": "mirror"}),
            },
            "optional": {
                "motion": ("DEPTHFLOW_MOTION",),  # Motion object
                "effects": ("DEPTHFLOW_EFFECTS",),  # DepthState object
            },
        }

    RETURN_TYPES = (
        "IMAGE",
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
        depth_map,
        animation_speed,
        input_fps,
        output_fps,
        num_frames,
        quality,
        ssaa,
        invert,
        tiling_mode,
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
        if depth_map.is_cuda:
            depth_map = depth_map.cpu().numpy()
        else:
            depth_map = depth_map.numpy()

        if image.ndim != 4:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if depth_map.ndim != 4:
            raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")

        if image.shape[0] != 1:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if depth_map.shape[0] != 1:
            raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")

        # Get width and height of images
        height, width = image.shape[1], image.shape[2]

        # Input the image and depthmap into the scene
        scene.input(image[0], depth=depth_map[0])


        # Alternatively, if your scene has a helper method:
        scene.inpaint(
            enable=True,
            black=False,
            limit=0.8
        )

        scene.zoom(
            intensity=1.5,
            smooth=False,
            loop=False
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
            quality=quality,
            ssaa=ssaa,
            scale=1.0,
            width=width,
            height=height,
            ratio=None,
            freewheel=True,
        )

        video = scene.get_accumulated_frames()
        scene.clear_frames()
        self.end_progress()

        # Normalize the video frames to [0, 1]
        video = video.float() / 255.0

        return (video,)
