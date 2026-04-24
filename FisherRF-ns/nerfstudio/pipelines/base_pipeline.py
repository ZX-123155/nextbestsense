# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import json
import cv2
import re

import random
import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.ros_utils import convertPose2Numpy

from nerfstudio.cameras.cameras import Cameras
from gsplat.project_gaussians import project_gaussians

from einops import repeat, reduce, rearrange

from tqdm import tqdm
import copy

# ROS related imports
import rospy
from std_msgs.msg import String
from gaussian_splatting.srv import NBVPoses, NBVPosesRequest, NBVPosesResponse, NBVResultRequest, NBVResultResponse, NBVResult
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest


def _ros_param_bool(name: str, default: bool) -> bool:
    """Read a ROS param as a real boolean, handling string values like 'true'/'false'."""
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)

import numpy as np


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.
        """

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    @abstractmethod
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""


class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        
        # ROS specific code
        rospy.init_node('nerf_pipeline', anonymous=True)
        rospy.loginfo("Starting the pipeline node!")
        self.touch_phase = False
        
        # import pdb; pdb.set_trace()

        # get random or fisher param
        self.view_selection_method = rospy.get_param('view_selection_method', 'fisher')

        # get views to add 
        self.views_to_add = int(rospy.get_param('views_to_add', 10)) # type: ignore
        self.nbv_start_step = int(rospy.get_param('nbv_start_step', 5000)) # type: ignore
        self.nbv_interval_steps = int(rospy.get_param('nbv_interval_steps', 1000)) # type: ignore
        self.nbv_max_rounds = int(rospy.get_param('nbv_max_rounds', self.views_to_add)) # type: ignore
        self.nbv_wait_timeout_s = float(rospy.get_param('nbv_wait_timeout_s', 300.0)) # type: ignore
        self.semantic_guided_nbv = _ros_param_bool('semantic_guided_nbv', True) # type: ignore
        self.nbv_bg_penalty = float(rospy.get_param('nbv_bg_penalty', 0.2)) # type: ignore
        self.nbv_score_weight = float(rospy.get_param('nbv_score_weight', rospy.get_param('nbv_bg_penalty', 0.3))) # type: ignore
        self.nbv_leaf_weight = float(rospy.get_param('nbv_leaf_weight', 0.25)) # type: ignore
        self.nbv_r_min = float(rospy.get_param('nbv_r_min', 0.2)) # type: ignore
        self.nbv_min_valid_candidates = int(rospy.get_param('nbv_min_valid_candidates', 2)) # type: ignore
        self.nbv_ig_history: List[Dict[str, Any]] = []
        
        self.touches_to_add = int(rospy.get_param('touches_to_add', 5)) # type: ignore
 
        self.added_views_so_far = 0
        self.added_touches_so_far = 0
        
        self.new_view_ready = False
        # create service to receive 
        self.continue_training_srv = rospy.Service('continue_training', Trigger, self.continue_training)
        
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device
    
    def continue_training(self, req: TriggerRequest) -> TriggerResponse:
        rospy.loginfo(f"Received request to continue training: {req}")
        self.new_view_ready = True
        
        # add new view to the training set
        res = TriggerResponse()
        res.success = True
        return res

    def _camera_scalar(self, value: Any, default: float) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.reshape(-1)[0].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_nbv_heatmap_dir(self) -> Optional[Path]:
        if hasattr(self, "_nbv_heatmap_dir") and self._nbv_heatmap_dir is not None:
            return self._nbv_heatmap_dir

        data_root = Path("/home/ras/NextBestSense/data")
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
        run_root: Optional[Path] = None

        if hasattr(self.datamanager, "train_dataparser_outputs"):
            image_filenames = getattr(self.datamanager.train_dataparser_outputs, "image_filenames", [])
            for image_path in image_filenames:
                try:
                    resolved = Path(str(image_path)).resolve()
                except Exception:
                    continue
                if data_root in resolved.parents:
                    relative_parts = resolved.relative_to(data_root).parts
                    if len(relative_parts) > 0 and date_pattern.match(relative_parts[0]):
                        run_root = data_root / relative_parts[0]
                        break

        if run_root is None and data_root.exists():
            dated_dirs = [d for d in data_root.iterdir() if d.is_dir() and date_pattern.match(d.name)]
            if len(dated_dirs) > 0:
                run_root = max(dated_dirs, key=lambda p: p.stat().st_mtime)

        if run_root is None:
            rospy.logwarn("Unable to resolve run date directory under /home/ras/NextBestSense/data; skipping NBV heatmap saving.")
            self._nbv_heatmap_dir = None
            return None

        heatmap_dir = run_root / "heatmap"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        self._nbv_heatmap_dir = heatmap_dir
        rospy.loginfo(f"NBV heatmaps will be saved to: {heatmap_dir}")
        return heatmap_dir

    def _build_alpha_blend_ig_heatmap(self, cam: Cameras, gaussian_values: torch.Tensor) -> Optional[np.ndarray]:
        if not hasattr(self.model, "means") or not hasattr(self.model, "opacities"):
            return None
        if not hasattr(self.model, "scales") or not hasattr(self.model, "quats"):
            return None

        means = getattr(self.model, "means").detach()
        opacities = torch.sigmoid(getattr(self.model, "opacities").detach()).squeeze(-1)
        scales = torch.exp(getattr(self.model, "scales").detach())
        quats = getattr(self.model, "quats").detach()
        quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-12)

        if means.ndim != 2 or means.shape[-1] != 3:
            return None
        if gaussian_values.numel() != means.shape[0] or opacities.numel() != means.shape[0]:
            return None

        height = int(round(self._camera_scalar(cam.height, 640.0)))
        width = int(round(self._camera_scalar(cam.width, 640.0)))
        fx = self._camera_scalar(cam.fx, 1.0)
        fy = self._camera_scalar(cam.fy, 1.0)
        cx = self._camera_scalar(cam.cx, width * 0.5)
        cy = self._camera_scalar(cam.cy, height * 0.5)

        if height <= 0 or width <= 0:
            return None

        c2w = cam.camera_to_worlds
        if not isinstance(c2w, torch.Tensor) or c2w.ndim != 3 or c2w.shape[0] == 0:
            return None

        c2w = c2w.to(device=means.device, dtype=means.dtype)
        R = c2w[0, :3, :3]
        T = c2w[0, :3, 3:4]
        R_edit = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=means.device, dtype=means.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=means.device, dtype=means.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv

        block_width = 16
        xys, depths, radii, _conics, _comp, _num_tiles_hit, _cov3d = project_gaussians(  # type: ignore
            means,
            scales,
            1,
            quats,
            viewmat[:3, :],
            fx,
            fy,
            cx,
            cy,
            height,
            width,
            block_width,
        )

        valid = (radii > 0) & torch.isfinite(depths) & torch.isfinite(xys).all(dim=-1)
        if not bool(valid.any()):
            return np.zeros((height, width), dtype=np.float32)

        valid_idx = torch.where(valid)[0]
        xy_valid = xys[valid_idx]
        u = torch.round(xy_valid[:, 0]).long().detach().cpu().numpy()
        v = torch.round(xy_valid[:, 1]).long().detach().cpu().numpy()
        depth = depths[valid_idx].detach().cpu().numpy()
        alpha = opacities[valid_idx].clamp(0.0, 1.0).detach().cpu().numpy()
        ig_vals = gaussian_values[valid_idx].detach().cpu().numpy()

        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_img):
            return np.zeros((height, width), dtype=np.float32)

        u = u[in_img]
        v = v[in_img]
        depth = depth[in_img]
        alpha = alpha[in_img]
        ig_vals = ig_vals[in_img]

        pix_id = v.astype(np.int64) * int(width) + u.astype(np.int64)
        order = np.lexsort((depth, pix_id))
        pix_id = pix_id[order]
        alpha = alpha[order]
        ig_vals = ig_vals[order]

        heat = np.zeros((height, width), dtype=np.float32)
        current_pix = -1
        transmittance = 1.0
        accum = 0.0

        for pid, a, ig in zip(pix_id, alpha, ig_vals):
            if pid != current_pix:
                if current_pix >= 0:
                    py = current_pix // width
                    px = current_pix % width
                    heat[py, px] = np.float32(accum)
                current_pix = int(pid)
                transmittance = 1.0
                accum = 0.0

            weight = transmittance * float(a)
            accum += weight * float(ig)
            transmittance *= (1.0 - float(a))

        if current_pix >= 0:
            py = current_pix // width
            px = current_pix % width
            heat[py, px] = np.float32(accum)

        return heat

    def _save_nbv_heatmap(
        self,
        heatmap: np.ndarray,
        round_idx: int,
        view_idx: int,
        is_touch: bool,
        variant: str = "raw",
        vis_low: Optional[float] = None,
        vis_high: Optional[float] = None,
    ) -> None:
        out_dir = self._resolve_nbv_heatmap_dir()
        if out_dir is None:
            return

        suffix = "_touch" if is_touch else ""
        base_name = f"nbv{round_idx:02d}_view{view_idx:02d}{suffix}_{variant}"

        npy_path = out_dir / f"{base_name}.npy"
        png_path = out_dir / f"{base_name}.png"

        np.save(npy_path, heatmap.astype(np.float32))

        heat_vis = np.log1p(np.clip(heatmap, a_min=0.0, a_max=None))
        pos_vals = heat_vis[heat_vis > 0]
        if pos_vals.size == 0:
            norm = np.zeros_like(heat_vis, dtype=np.uint8)
        else:
            low = float(np.percentile(pos_vals, 5.0)) if vis_low is None else float(vis_low)
            high = float(np.percentile(pos_vals, 99.7)) if vis_high is None else float(vis_high)
            if high - low < 1e-12:
                scaled = np.zeros_like(heat_vis, dtype=np.float32)
            else:
                scaled = np.clip((heat_vis - low) / (high - low), 0.0, 1.0)
            scaled = np.power(scaled, 0.65)
            norm = (scaled * 255.0).astype(np.uint8)

        color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(png_path), color)

    def _save_selected_view_heatmaps(
        self,
        cam: Cameras,
        raw_values: torch.Tensor,
        roi_gaussian_mask: Optional[torch.Tensor],
        round_idx: int,
        view_idx: int,
        is_touch: bool,
    ) -> None:
        raw_heatmap = self._build_alpha_blend_ig_heatmap(cam, raw_values)
        if raw_heatmap is None:
            return

        if roi_gaussian_mask is not None and roi_gaussian_mask.numel() == raw_values.numel():
            roi_values = roi_gaussian_mask.to(dtype=raw_values.dtype)
            roi_prob_heatmap = self._build_alpha_blend_ig_heatmap(cam, roi_values)
            if roi_prob_heatmap is not None:
                roi_pixel_mask = roi_prob_heatmap >= 0.5
                # Keep ROI unchanged and only penalize background in visualization.
                weighted_heatmap = raw_heatmap * np.where(roi_pixel_mask, 1.0, 0.3).astype(np.float32)
            else:
                weighted_heatmap = raw_heatmap * np.float32(0.3)
        else:
            weighted_heatmap = raw_heatmap * np.float32(0.3)

        vis_ref = raw_heatmap if raw_heatmap is not None else weighted_heatmap
        assert vis_ref is not None
        ref_vis = np.log1p(np.clip(vis_ref, a_min=0.0, a_max=None))
        ref_pos = ref_vis[ref_vis > 0]
        if ref_pos.size > 0:
            vis_low = float(np.percentile(ref_pos, 5.0))
            vis_high = float(np.percentile(ref_pos, 99.7))
        else:
            vis_low = 0.0
            vis_high = 1.0

        self._save_nbv_heatmap(
            raw_heatmap,
            round_idx,
            view_idx,
            is_touch,
            variant="raw",
            vis_low=vis_low,
            vis_high=vis_high,
        )

        self._save_nbv_heatmap(
            weighted_heatmap,
            round_idx,
            view_idx,
            is_touch,
            variant="weighted",
            vis_low=vis_low,
            vis_high=vis_high,
        )

    def _compose_leafreplace_raw_heatmap(
        self,
        raw_heatmap: np.ndarray,
        leaf_heatmap: np.ndarray,
    ) -> np.ndarray:
        """Replace leaf-region pixels in raw heatmap with leaf heatmap values."""
        composed = raw_heatmap.copy()
        leaf_pos = leaf_heatmap[leaf_heatmap > 0]
        if leaf_pos.size == 0:
            return composed
        threshold = float(np.percentile(leaf_pos, 10.0))
        leaf_mask = leaf_heatmap > threshold
        composed[leaf_mask] = leaf_heatmap[leaf_mask]
        return composed

    def _save_real_mask_weighted_uncertainty(
        self,
        new_view_idx: int,
        rgb_weight: float,
        depth_weight: float,
        round_idx: int,
    ) -> None:
        """Use real captured mask to generate post-move weighted uncertainty maps."""
        if not hasattr(self.model, "render_uncertainty_rgb_depth"):
            return
        try:
            cam, batch = self.datamanager.get_cam_data_from_idx(new_view_idx)  # type: ignore
            training_cameras: List[Cameras] = []
            for idx in self.datamanager.get_current_views():  # type: ignore
                tcam, _ = self.datamanager.get_cam_data_from_idx(idx)  # type: ignore
                training_cameras.append(tcam)

            unc_maps = self.model.render_uncertainty_rgb_depth(  # type: ignore
                training_cameras,
                [cam],
                rgb_weight=rgb_weight,
                depth_weight=depth_weight,
                real_object_masks=[batch["mask"]] if "mask" in batch else None,
                real_semantic_masks=[batch["semantic_mask"]] if "semantic_mask" in batch else None,
                apply_real_mask_weighting=True,
            )
            if unc_maps is None or len(unc_maps) == 0:
                return

            uncertainty_raw = unc_maps[0].detach().float().cpu().numpy()
            uncertainty_raw_log = np.log1p(np.clip(uncertainty_raw, a_min=0.0, a_max=None))
            h, w = uncertainty_raw.shape[:2]
            uncertainty_raw_heatmap = uncertainty_raw_log.copy()
            uncertainty_weighted = uncertainty_raw_heatmap.copy()
            bg_scale = np.float32(float(getattr(self.model.config, "uncertainty_bg_weight", self.nbv_score_weight)))  # type: ignore
            leaf_scale = np.float32(float(getattr(self.model.config, "uncertainty_leaf_weight", 0.25)))  # type: ignore
            fruit_scale = np.float32(float(getattr(self.model.config, "uncertainty_fruit_weight", 1.0)))  # type: ignore

            fruit_gate = np.zeros((h, w), dtype=np.float32)
            leaf_gate = np.zeros((h, w), dtype=np.float32)
            if "semantic_mask" in batch:
                sem_np = batch["semantic_mask"].detach().cpu().numpy()
                if sem_np.ndim == 3:
                    sem_np = sem_np[..., 0]
                if sem_np.shape != (h, w):
                    sem_np = cv2.resize(sem_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                fruit_value = float(getattr(self.model.config, "fruit_mask_value", 255))  # type: ignore
                leaf_value = float(getattr(self.model.config, "leaf_mask_value", 128))  # type: ignore
                tol = float(getattr(self.model.config, "semantic_mask_tolerance", 32))  # type: ignore
                fruit_gate = (np.abs(sem_np.astype(np.float32) - fruit_value) <= tol).astype(np.float32)
                leaf_gate = (np.abs(sem_np.astype(np.float32) - leaf_value) <= tol).astype(np.float32)
                leaf_gate = leaf_gate * (1.0 - fruit_gate)

            bg_gate = np.clip(1.0 - fruit_gate - leaf_gate, 0.0, 1.0).astype(np.float32)
            uncertainty_fruit_weighted = uncertainty_raw_heatmap * fruit_gate * fruit_scale
            uncertainty_leaf_weighted = uncertainty_raw_heatmap * leaf_gate * leaf_scale
            # Apply semantic-region weights in log space so semantic-weighted map diverges from raw map.
            uncertainty_weighted = uncertainty_raw_heatmap * (
                fruit_scale * fruit_gate + leaf_scale * leaf_gate + bg_scale * bg_gate
            )

            ref_vis = uncertainty_raw_heatmap
            ref_pos = ref_vis[ref_vis > 0]
            if ref_pos.size > 0:
                vis_low = float(np.percentile(ref_pos, 5.0))
                vis_high = float(np.percentile(ref_pos, 99.7))
            else:
                vis_low = 0.0
                vis_high = 1.0

            self._save_nbv_heatmap(np.expm1(np.clip(uncertainty_raw_heatmap, a_min=0.0, a_max=None)), round_idx, new_view_idx, False, "uncertainty_raw_heatmap_realmask", vis_low, vis_high)
            self._save_nbv_heatmap(np.expm1(np.clip(uncertainty_weighted, a_min=0.0, a_max=None)), round_idx, new_view_idx, False, "uncertainty_semantic_weighted_heatmap_realmask", vis_low, vis_high)
            self._save_nbv_heatmap(np.expm1(np.clip(uncertainty_leaf_weighted, a_min=0.0, a_max=None)), round_idx, new_view_idx, False, "uncertainty_leaf_weighted_realmask", vis_low, vis_high)
            self._save_nbv_heatmap(np.expm1(np.clip(uncertainty_fruit_weighted, a_min=0.0, a_max=None)), round_idx, new_view_idx, False, "uncertainty_fruit_weighted_realmask", vis_low, vis_high)
        except Exception as exc:
            rospy.logwarn(f"Failed to generate real-mask weighted uncertainty maps: {exc}")
    
    
    def fisher_single_view(self, training_views: List[int], candidate_views: List[np.ndarray],
                           rgb_weight=1.0, depth_weight=1.0, camera_info=None) -> Tuple[Optional[int], List[float]]:
        # construct initial Hessian matrix from the current training data
        H_train = None
        sample_cam: Cameras = None # type: ignore
        training_cameras: List[Cameras] = []
        scale_factor = self.datamanager.train_dataparser_outputs.dataparser_scale  # type: ignore
        
        for view in training_views:
            # get full camera from view idx
            cam, batch = self.datamanager.get_cam_data_from_idx(view)
            if sample_cam is None:
                sample_cam = cam
            training_cameras.append(cam)
                
            cur_H: torch.tensor = self.compute_hessian(
                cam,
                rgb_weight,
                depth_weight,
                use_object_mask=self.semantic_guided_nbv,
                object_mask_weight=self.nbv_bg_penalty,
            ) # type: ignore
            
            if H_train is None:
                H_train = cur_H
            else:
                H_train += cur_H
                
        H_train = H_train.to(self.device) # type: ignore
        reg_lambda = 1e-6
        I_train = torch.reciprocal(H_train + reg_lambda)
        
        acq_scores = torch.zeros(len(candidate_views)) # type: ignore
        ig_view_metrics: List[Dict[str, Any]] = []
        candidate_state: List[Dict[str, Any]] = []
        rospy.loginfo("Hessian computed! Computing NBV...")
        
        # go through each candidate camera and calculate the acq score
        for idx, view_pos in enumerate(tqdm(candidate_views, desc="Computing acq scores")):
            # copy sample_cam and update the pose
            cam = copy.deepcopy(sample_cam)
            if camera_info is not None:
                cam.fx[0][0] = camera_info['fx']
                cam.fy[0][0] = camera_info['fy']
                cam.cx[0][0] = camera_info['cx']
                cam.cy[0][0] = camera_info['cy']
                cam.width[0][0] = camera_info['w']
                cam.height[0][0] = camera_info['h']
                
                self.dt_cam = cam
                
            cam.metadata = None
            
            view_pos = torch.from_numpy(view_pos.astype(np.float32)).to(self.device)
            view_pos[:3, 3] *= scale_factor
            cam.camera_to_worlds = view_pos[:3, :4].unsqueeze(0)
            
            # if camera_info is not None, it is a touch view
            is_touch = camera_info is not None
            use_object_mask = self.semantic_guided_nbv or is_touch
            
            cur_H: torch.tensor = self.compute_hessian(
                cam,
                rgb_weight,
                depth_weight,
                is_touch=is_touch,
                use_object_mask=use_object_mask,
                object_mask_weight=self.nbv_bg_penalty,
            ) # type: ignore
            I_acq = cur_H
            contrib = I_acq * I_train
            acq_scores[idx] = torch.sum(contrib).item()

            ig_roi = float(torch.sum(contrib).item())
            ig_leaf = 0.0
            ig_bg = 0.0
            roi_mask_tensor = None
            if use_object_mask and hasattr(self.model, "sam_mask") and getattr(self.model, "sam_mask") is not None:
                sam_masks_prob = torch.sigmoid(getattr(self.model, "sam_mask").detach())
                if sam_masks_prob.ndim > 1:
                    sam_masks_prob = sam_masks_prob.squeeze(-1)
                roi_mask_tensor = sam_masks_prob > 0.5
                has_semantic_parts = hasattr(self.model, "gauss_params") and (
                    "sam_mask_fruit" in getattr(self.model, "gauss_params")
                    and "sam_mask_leaf" in getattr(self.model, "gauss_params")
                )
                if has_semantic_parts:
                    fruit_prob = torch.sigmoid(getattr(self.model, "gauss_params")["sam_mask_fruit"].detach()).squeeze(-1)
                    leaf_prob = torch.sigmoid(getattr(self.model, "gauss_params")["sam_mask_leaf"].detach()).squeeze(-1)
                    fruit_mask = fruit_prob > 0.5
                    leaf_mask = (leaf_prob > 0.5) & (~fruit_mask)
                    bg_mask = ~(fruit_mask | leaf_mask)
                    ig_fruit = float(torch.sum(contrib[fruit_mask]).item()) if bool(fruit_mask.any()) else 0.0
                    ig_leaf = float(torch.sum(contrib[leaf_mask]).item()) if bool(leaf_mask.any()) else 0.0
                    ig_bg = float(torch.sum(contrib[bg_mask]).item()) if bool(bg_mask.any()) else 0.0
                    ig_roi = ig_fruit + float(self.nbv_leaf_weight) * ig_leaf
                    r_ratio = ig_fruit / (ig_fruit + ig_leaf + ig_bg + 1e-12)
                else:
                    bg_mask = ~roi_mask_tensor
                    if bool(roi_mask_tensor.any()):
                        ig_roi = float(torch.sum(contrib[roi_mask_tensor]).item())
                    else:
                        ig_roi = 0.0
                    if bool(bg_mask.any()):
                        ig_bg = float(torch.sum(contrib[bg_mask]).item())
                    else:
                        ig_bg = 0.0
                    r_ratio = ig_roi / (ig_roi + ig_bg + 1e-12)
            else:
                r_ratio = ig_roi / (ig_roi + ig_bg + 1e-12)
            ig_view_metrics.append(
                {
                    "view_idx": int(idx),
                    "ig_roi": ig_roi,
                    "ig_leaf": ig_leaf,
                    "ig_bg": ig_bg,
                    "r": float(r_ratio),
                    "selected": False,
                }
            )
            candidate_state.append(
                {
                    "cam": cam,
                    "contrib": contrib,
                    "roi_mask": roi_mask_tensor,
                    "view_pos": view_pos,
                    "is_touch": is_touch,
                }
            )

        # ROI-prioritized scoring with background penalty:
        # 1) Filter candidates with low ROI ratio (R < nbv_r_min)
        # 2) Score with S = (1 - nbv_score_weight) * IG_roi + nbv_score_weight * IG_bg
        filtered_penalty = -1e12
        semantic_scores: List[float] = []
        for item in ig_view_metrics:
            ig_roi = float(item["ig_roi"])
            ig_bg = float(item["ig_bg"])
            r_ratio = float(item["r"])

            if r_ratio < self.nbv_r_min:
                score = filtered_penalty
            else:
                score = float((1.0 - self.nbv_score_weight) * ig_roi + self.nbv_score_weight * ig_bg)

            item["score"] = score
            semantic_scores.append(score)

        valid_candidates = [item for item in ig_view_metrics if float(item["r"]) >= float(self.nbv_r_min)]
        if len(valid_candidates) < int(self.nbv_min_valid_candidates):
            rospy.logwarn(
                "NBV round skipped: valid candidates with R >= %.4f are %d (< %d)",
                float(self.nbv_r_min),
                len(valid_candidates),
                int(self.nbv_min_valid_candidates),
            )
            self.nbv_ig_history.append(
                {
                    "round": int(self.added_touches_so_far) + 1 if camera_info is not None else int(self.added_views_so_far) + 1,
                    "is_touch": bool(camera_info is not None),
                    "views": ig_view_metrics,
                    "selected": None,
                    "valid_candidates": len(valid_candidates),
                }
            )
            return None, []

        acq_scores = torch.tensor(
            semantic_scores,
            device=cur_H.device if isinstance(cur_H, torch.Tensor) else None,
            dtype=torch.float32,
        )
        top_idx = int(torch.argmax(acq_scores).item())
        
        selected_idx = int(top_idx)
        print('Selected views:', selected_idx)

        for item in ig_view_metrics:
            item["selected"] = bool(item["view_idx"] == selected_idx)

        selected_state = candidate_state[selected_idx]
        selected_cam = selected_state["cam"]
        selected_roi_mask = selected_state["roi_mask"]

        if hasattr(self.model, "render_uncertainty_rgb_depth") and callable(getattr(self.model, "render_uncertainty_rgb_depth")):
            try:
                unc_maps = self.model.render_uncertainty_rgb_depth(
                    training_cameras,
                    [selected_cam],
                    rgb_weight=rgb_weight,
                    depth_weight=depth_weight,
                )  # type: ignore
                if unc_maps is not None and len(unc_maps) > 0:
                    uncertainty_raw = unc_maps[0].detach().float().cpu().numpy()
                    uncertainty_raw_log = np.log1p(np.clip(uncertainty_raw, a_min=0.0, a_max=None))
                    bg_scale = np.float32(float(getattr(self.model.config, "uncertainty_bg_weight", self.nbv_score_weight)))  # type: ignore
                    leaf_scale = np.float32(float(getattr(self.model.config, "uncertainty_leaf_weight", 0.25)))  # type: ignore
                    fruit_scale = np.float32(float(getattr(self.model.config, "uncertainty_fruit_weight", 1.0)))  # type: ignore

                    # 1) Base: suppress non-ROI (background) with bg_scale; ROI remains full scale.
                    if selected_roi_mask is not None:
                        roi_values = selected_roi_mask.to(dtype=torch.float32)
                        roi_prob_heatmap = self._build_alpha_blend_ig_heatmap(selected_cam, roi_values)
                        if roi_prob_heatmap is not None:
                            roi_gate = np.clip(roi_prob_heatmap.astype(np.float32), 0.0, 1.0)
                            base = uncertainty_raw_log * (bg_scale + (np.float32(1.0) - bg_scale) * roi_gate)
                        else:
                            base = uncertainty_raw_log * bg_scale
                    else:
                        base = uncertainty_raw_log * bg_scale

                    uncertainty_raw_heatmap = base.copy()
                    uncertainty_weighted = base.copy()
                    uncertainty_leaf_weighted = base.copy()
                    uncertainty_fruit_weighted = base.copy()

                    # 2) On `base`, apply per-class multipliers (fruit / leaf / semantic background).
                    has_semantic_parts = hasattr(self.model, "gauss_params") and (
                        "sam_mask_fruit" in getattr(self.model, "gauss_params")
                        and "sam_mask_leaf" in getattr(self.model, "gauss_params")
                    )
                    if has_semantic_parts:
                        fruit_prob_heatmap = None
                        leaf_prob_heatmap = None
                        # Prefer camera-rendered semantic masks for cleaner, contiguous shapes.
                        try:
                            if hasattr(self.model, "get_outputs_for_camera"):
                                sem_out = self.model.get_outputs_for_camera(selected_cam)  # type: ignore
                            else:
                                sem_out = self.model.get_outputs(selected_cam)  # type: ignore
                            if "fruit_mask" in sem_out and "leaf_mask" in sem_out:
                                fruit_prob_heatmap = torch.sigmoid(sem_out["fruit_mask"].detach()).float().cpu().numpy()[..., 0]
                                leaf_prob_heatmap = torch.sigmoid(sem_out["leaf_mask"].detach()).float().cpu().numpy()[..., 0]
                        except Exception:
                            fruit_prob_heatmap = None
                            leaf_prob_heatmap = None

                        # Fallback to gaussian-space alpha-blend approximation.
                        if fruit_prob_heatmap is None or leaf_prob_heatmap is None:
                            fruit_values = torch.sigmoid(getattr(self.model, "gauss_params")["sam_mask_fruit"].detach()).squeeze(-1)  # type: ignore
                            leaf_values = torch.sigmoid(getattr(self.model, "gauss_params")["sam_mask_leaf"].detach()).squeeze(-1)  # type: ignore
                            fruit_prob_heatmap = self._build_alpha_blend_ig_heatmap(selected_cam, fruit_values.to(dtype=torch.float32))
                            leaf_prob_heatmap = self._build_alpha_blend_ig_heatmap(selected_cam, leaf_values.to(dtype=torch.float32))

                        if fruit_prob_heatmap is not None and leaf_prob_heatmap is not None:
                            # Use soft per-pixel semantic probabilities instead of hard thresholding,
                            # so weighted uncertainty keeps continuous fruit/leaf shapes.
                            fruit_gate = np.clip(fruit_prob_heatmap.astype(np.float32), 0.0, 1.0)
                            leaf_gate_raw = np.clip(leaf_prob_heatmap.astype(np.float32), 0.0, 1.0)
                            # Smooth and densify gates for better contiguous shape visualization.
                            fruit_gate = cv2.GaussianBlur(fruit_gate, (0, 0), sigmaX=1.2, sigmaY=1.2)
                            leaf_gate_raw = cv2.GaussianBlur(leaf_gate_raw, (0, 0), sigmaX=1.2, sigmaY=1.2)
                            morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            fruit_bin = (fruit_gate > 0.25).astype(np.uint8)
                            leaf_bin = (leaf_gate_raw > 0.25).astype(np.uint8)
                            fruit_bin = cv2.morphologyEx(fruit_bin, cv2.MORPH_CLOSE, morph_kernel)
                            leaf_bin = cv2.morphologyEx(leaf_bin, cv2.MORPH_CLOSE, morph_kernel)
                            fruit_gate = np.clip(fruit_gate * fruit_bin.astype(np.float32), 0.0, 1.0)
                            leaf_gate_raw = np.clip(leaf_gate_raw * leaf_bin.astype(np.float32), 0.0, 1.0)
                            # Fruit has priority on overlap; leaf uses remaining probability mass.
                            leaf_gate = np.clip(leaf_gate_raw * (1.0 - fruit_gate), 0.0, 1.0)
                            # Neither fruit nor leaf: semantic background (optional extra down-weight in semantic map).
                            sem_bg = np.clip(1.0 - fruit_gate - leaf_gate, 0.0, 1.0).astype(np.float32)
                            # raw_heatmap: fruit+leaf both scaled by fruit_scale; other pixels keep `base` (already ROI/BG suppressed)
                            uncertainty_raw_heatmap = base * (fruit_scale * (fruit_gate + leaf_gate) + sem_bg)
                            # semantic_weighted: fruit * fruit_w, leaf * leaf_w, semantic bg * bg_w
                            uncertainty_weighted = base * (fruit_scale * fruit_gate + leaf_scale * leaf_gate + bg_scale * sem_bg)
                            uncertainty_fruit_weighted = base * fruit_gate * fruit_scale
                            uncertainty_leaf_weighted = base * leaf_gate * leaf_scale

                    ref_vis = uncertainty_raw_log
                    ref_pos = ref_vis[ref_vis > 0]
                    if ref_pos.size > 0:
                        vis_low = float(np.percentile(ref_pos, 5.0))
                        vis_high = float(np.percentile(ref_pos, 99.7))
                    else:
                        vis_low = 0.0
                        vis_high = 1.0

                    round_idx = int(self.added_touches_so_far) + 1 if camera_info is not None else int(self.added_views_so_far) + 1
                    self._save_nbv_heatmap(
                        np.expm1(np.clip(uncertainty_raw_heatmap, a_min=0.0, a_max=None)),
                        round_idx,
                        selected_idx,
                        bool(camera_info is not None),
                        variant="uncertainty_raw_heatmap",
                        vis_low=vis_low,
                        vis_high=vis_high,
                    )
                    self._save_nbv_heatmap(
                        np.expm1(np.clip(uncertainty_weighted, a_min=0.0, a_max=None)),
                        round_idx,
                        selected_idx,
                        bool(camera_info is not None),
                        variant="uncertainty_semantic_weighted_heatmap",
                        vis_low=vis_low,
                        vis_high=vis_high,
                    )
                    self._save_nbv_heatmap(
                        np.expm1(np.clip(uncertainty_leaf_weighted, a_min=0.0, a_max=None)),
                        round_idx,
                        selected_idx,
                        bool(camera_info is not None),
                        variant="uncertainty_leaf_weighted",
                        vis_low=vis_low,
                        vis_high=vis_high,
                    )
                    self._save_nbv_heatmap(
                        np.expm1(np.clip(uncertainty_fruit_weighted, a_min=0.0, a_max=None)),
                        round_idx,
                        selected_idx,
                        bool(camera_info is not None),
                        variant="uncertainty_fruit_weighted",
                        vis_low=vis_low,
                        vis_high=vis_high,
                    )
            except Exception as exc:
                rospy.logwarn(f"Failed to render/save NBV model uncertainty heatmap: {exc}")

        self.nbv_ig_history.append(
            {
                "round": int(self.added_views_so_far) + 1,
                "is_touch": bool(camera_info is not None),
                "views": ig_view_metrics,
            }
        )

        return selected_idx, acq_scores.tolist()
        
    
    def view_selection(self, training_views: List[int], candidate_views: List[np.ndarray], option='random',
                       rgb_weight=1.0, depth_weight=1.0,
                       camera_info=None) -> Tuple[Optional[int], List[float]]: # type: ignore
        """
        Args:
            training_views (List[int]): List of training views by index.
            candidate_views (List[np.ndarray]): Candidate views to select from -- a list of poses.
            option (str, optional): The method of view selection. Either random or fisher. Defaults to 'random'.
            rgb_weight (float, optional): RGB weight for Fisherrf. Defaults to 1.0.
            depth_weight (float, optional): Depth weight for Fisherrf. Defaults to 1.0.
            is_touch (bool, optional): whether the view is a touch view. Defaults to False.

        Returns:
            Tuple[int, List[float]]: _description_
        """
        if option == 'random':
            # construct scores based on number of candidate views
            scores = [random.random() for _ in range(len(candidate_views))]
            # get argmax of scores
            max_idx = scores.index(max(scores))
            return max_idx, scores
        
        elif option == "fisher":
            # use fisher information to select the next view
            selected_view, acq_scores = self.fisher_single_view(training_views, candidate_views, rgb_weight, depth_weight,
                                                                camera_info=camera_info) # type: ignore
            return selected_view, acq_scores # type: ignore 

        elif option == "fisher-multi-view":
            # batch fisher information to select the next views
            scores = [random.random() for _ in range(len(candidate_views))]
            # get argmax of scores
            max_idx = scores.index(max(scores))
            return max_idx, scores
        else:
            # otherwise, select the first view in list. Not recommended.
            max_idx = 0
            scores = [100.0 - i for i in range(len(candidate_views))]
            return max_idx, scores
        
    def call_get_nbv_poses(self) -> List[np.ndarray]:
        rospy.wait_for_service('get_poses')
        rospy.loginfo("Calling service to get NBV poses")
        final_poses: List[np.ndarray] = []
        
        # make service call to get the list of avail poses
        try:
            get_nbv_poses = rospy.ServiceProxy('get_poses', NBVPoses)
            req = NBVPosesRequest()
            response: NBVPosesResponse = get_nbv_poses(req)
            if response.success:
                rospy.loginfo(f"Response Message: {response.message}")
                poses: List[PoseStamped] = response.poses
                ns_poses: List[np.ndarray] = [convertPose2Numpy(pose) for pose in poses] # type: ignore
                # transform to play nice with nerfstudio
                for pose in ns_poses:
                    new_pose = pose
                    new_pose[0:3, 1:3] *= -1
                    final_poses.append(new_pose)
            else: 
                rospy.loginfo("Failed to get poses. Adding no views")
            
        except rospy.ROSException as e:
            rospy.loginfo(f"Service call failed: {e}")
            
        return final_poses
    
    def send_nbv_scores(self, scores) -> bool:
        rospy.wait_for_service('receive_nbv_scores')
        poses = []
        
        # make service call to get the list of avail poses
        try:
            receive_nbv_scores_service = rospy.ServiceProxy('receive_nbv_scores', NBVResult)
            req = NBVResultRequest()
            req.scores = list(scores)
            response: NBVPosesResponse = receive_nbv_scores_service(req)
            
            if response.success:
                rospy.loginfo(f"Response Message: {response.message}")
                return True
            else: 
                rospy.loginfo("Failed send NBV scores. Adding no views")
            
        except rospy.ROSException as e:
            rospy.loginfo(f"Service call failed: {e}")
        return False
    
    def run_nbv(self, rgb_weight=1.0, depth_weight=1.0, is_touch=False, touch_poses=None,
                camera_info=None):
        if not is_touch:
            avail_views = self.call_get_nbv_poses()
        else:
            avail_views = touch_poses

        if avail_views is None or len(avail_views) == 0:
            rospy.logwarn("No available candidate views for NBV.")
            return None, []

        rospy.loginfo("Selecting new view for training")
        current_views_idxs = self.datamanager.get_current_views()
        
        next_view, acq_scores = self.view_selection(current_views_idxs, avail_views, option=self.view_selection_method, # type: ignore
                                                rgb_weight=rgb_weight, depth_weight=depth_weight, camera_info=camera_info) # type: ignore
        
        return next_view, acq_scores


    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        
        ray_bundle, batch = self.datamanager.next_train(step)
        
        self.model.lift_depths_to_3d(ray_bundle, batch) # type: ignore
        self.model.camera = ray_bundle # type: ignore
        if not self.touch_phase:
            model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        else: 
            # touch phase
            metrics_dict = self.model.get_metrics_dict({}, batch)
            loss_dict = self.model.get_loss_dict({}, batch, metrics_dict)
            model_outputs = {}
            model_outputs['depth'] = metrics_dict['depth']
        is_nbv_step = step >= self.nbv_start_step and (step - self.nbv_start_step) % self.nbv_interval_steps == 0
        if is_nbv_step:
            if self.added_views_so_far < self.nbv_max_rounds:
                # add views to the training set if we can
                rgb_weight = float(getattr(self.model.config, "rgb_uncertainty_weight", 0.0)) # type: ignore
                depth_weight = float(getattr(self.model.config, "depth_uncertainty_weight", 1.0)) # type: ignore
                next_view, acq_scores = self.run_nbv(rgb_weight=rgb_weight, depth_weight=depth_weight, is_touch=False)

                if next_view is None or len(acq_scores) == 0:
                    rospy.logwarn("NBV skipped at step {} due to empty candidate set/scores".format(step))
                    return model_outputs, loss_dict, metrics_dict
                
                # send acquired scores in ROS
                success = self.send_nbv_scores(acq_scores)
                
                rate = rospy.Rate(1)  # 1 Hz    
                wait_start = time()
                while not rospy.is_shutdown() and not self.new_view_ready:
                    rospy.loginfo("Waiting for Robot Node to Be Done...")
                    if time() - wait_start > self.nbv_wait_timeout_s:
                        rospy.logwarn("NBV wait timed out after {}s; continuing training.".format(self.nbv_wait_timeout_s))
                        break
                    rate.sleep()
                rospy.loginfo("GS taking new view!")
                if success and self.new_view_ready:
                    self.datamanager.add_new_view(next_view) # type: ignore
                    self.model.camera_optimizer.add_camera() # type: ignore
                    print("Added new view succesfully.")
                    try:
                        new_view_idx = len(self.datamanager.get_current_views()) - 1  # type: ignore
                        self._save_real_mask_weighted_uncertainty(
                            new_view_idx=new_view_idx,
                            rgb_weight=rgb_weight,
                            depth_weight=depth_weight,
                            round_idx=int(self.added_views_so_far) + 1,
                        )
                    except Exception as exc:
                        rospy.logwarn(f"Failed post-move real-mask uncertainty visualization: {exc}")
                    # new view is not ready now
                    self.new_view_ready = False
                    self.added_views_so_far += 1

            else:
                if self.added_touches_so_far < self.touches_to_add:
                    # add new touch!! 
                    self.touch_phase = True
                    rospy.loginfo("Adding new touch view!")
                    
                    # for now, get from the list of touches
                    touch_data_dir = "/home/user/NextBestSense/data/ridge_touch/touch"
                    
                    with open(f'{touch_data_dir}/transforms.json', "r") as f:
                        data = json.load(f)
                        
                    frames = data['frames']
                    
                    poses = []
                    diff = 0.25  # 25 cm away from the touch in the negative z direction
                    orig_touch_poses = []
                    for frame in frames:
                        pose = frame['transformation']
                        pose_np = np.array(pose).reshape(4, 4)
                        # add poses along the z in the negative direction
                        z = pose_np[0:3, 2]
                        
                        orig_pose = pose_np.copy()
                        orig_pose[0:3, 1:3] *= -1
                        orig_touch_poses.append(orig_pose)
                        
                        # update the position in pose_np
                        pose_np[0:3, 3] -= (diff * z)
                        pose_np[0:3, 1:3] *= -1
                        poses.append(pose_np)
                        
                    camera_info  = {
                        'fx': 200,
                        'fy': 200,
                        'cx': 320,
                        'cy': 320,
                    }
                    camera_info['h'] = data['h']
                    camera_info['w'] = data['w']
                        
                    # get next best touch
                    rgb_weight = float(getattr(self.model.config, "rgb_uncertainty_weight", 0.0)) # type: ignore
                    depth_weight = float(getattr(self.model.config, "depth_uncertainty_weight", 1.0)) # type: ignore
                    next_touch, acq_scores = self.run_nbv(rgb_weight=rgb_weight, depth_weight=depth_weight, is_touch=True, 
                                                        touch_poses=poses, camera_info=camera_info)
                    if next_touch is None or len(acq_scores) == 0:
                        rospy.logwarn("Touch NBV skipped due to insufficient valid candidates")
                        return model_outputs, loss_dict, metrics_dict
                    
                    # get depth of the touch
                    touch_file_path = frames[next_touch]['file_path']
                    touch_pose = poses[next_touch]
                    touch_file_path = touch_file_path.split('/')[-1]
                    touch_file_path = touch_file_path.split('.')[0]
                    touch_file_path = f"{touch_data_dir}/{touch_file_path}_zmap.png"
                    
                    # read in depth
                    depth = cv2.imread(touch_file_path, cv2.IMREAD_UNCHANGED) / 1000.0
                    depth = depth / 1000.0
                    
                    # set low values to zero
                    depth[depth < 0.005] = 0.0
                    
                    # we have computed the best touch, now add it
                    # real surface is higher than current surface: add the gaussians. no need to squash
                    # start from the touch pose and slowly move the camera to the actual touch pose
                    is_real_surface_lower = True
                    
                    # if trajectory is stored, break up trajectory into discrete sections and prune Gaussians
                    if is_real_surface_lower:
                        import pdb; pdb.set_trace()
                        ros_touch_pose = poses[next_touch]
                        ros_touch_pose[0:3, 1:3] *= -1
                        z = ros_touch_pose[0:3, 2]
                        top_position = ros_touch_pose[0:3, 3]
                        bottom_position = top_position + (0.25 * z)
                        
                        # prune the Gaussians in the cylinder from n meters out to the touch
                        self.model.prune_gaussians(top_position, bottom_position, 0.03) # type: ignore
                        
                    sample_cam, _ = self.datamanager.get_cam_data_from_idx(0) # type: ignore
                    # copy the camera
                    dt_cam = copy.deepcopy(sample_cam)
                    dt_cam.metadata = None 
                    dt_cam.cx[0][0] = camera_info['cx']
                    dt_cam.cy[0][0] = camera_info['cy']
                    dt_cam.fx[0][0] = camera_info['fx']
                    dt_cam.fy[0][0] = camera_info['fy']
                    dt_cam.width = 640
                    dt_cam.height = 640
                    
                    touch_pose_torch = torch.from_numpy(touch_pose.astype(np.float32)).to(self.device)
                    dt_cam.camera_to_worlds = touch_pose_torch[:3, :4].unsqueeze(0)
                    depth = torch.tensor(depth).to(self.device)
                    self.model.add_touch_cam(dt_cam, depth) # type: ignore
                    
                    # now add the Gaussians for touch! These Gaussians are less likely to be culled as they are reasonably close to the surface
                    # self.model.add_touch_gaussians(touch_pose, camera_info, depth) # type: ignore
                    self.added_touches_so_far += 1
                    
                    # add to touch dataset with dt cam and provided depth
                    # self.model.add_touch_cam(self.dt_cam, depth) # type: ignore
                    
                    
                    # add a new touch camera to GS
                    
                    # update all depth maps with the touch depth
                    
        return model_outputs, loss_dict, metrics_dict
    
    
    def compute_hessian(self, ray_bundle, rgb_weight, depth_weight, is_touch=False, use_object_mask=False, object_mask_weight=0.3):
        if hasattr(self.model, 'compute_diag_H_rgb_depth') and callable(getattr(self.model, 'compute_diag_H_rgb_depth')):
            # compute the Hessian
            H_info_rgb = self.model.compute_diag_H_rgb_depth(
                ray_bundle,
                compute_rgb_H=True,
                is_touch=is_touch,
                apply_object_mask=use_object_mask,
                object_mask_weight=object_mask_weight,
            ) # type: ignore
            H_info_rgb['H'] = [p * rgb_weight for p in H_info_rgb['H']]
            H_per_gaussian = sum([reduce(p, "n ... -> n", "sum") for p in H_info_rgb['H']])
            
            H_info_depth = self.model.compute_diag_H_rgb_depth(
                ray_bundle,
                compute_rgb_H=False,
                is_touch=is_touch,
                apply_object_mask=use_object_mask,
                object_mask_weight=object_mask_weight,
            ) # type: ignore
            H_info_depth['H'] = [p * depth_weight for p in H_info_depth['H']]
            H_per_gaussian += sum([reduce(p, "n ... -> n", "sum") for p in H_info_depth['H']])
            return H_per_gaussian   

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
            # disable=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    raise NotImplementedError("Saving images is not implemented yet")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
