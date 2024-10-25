# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Model for InstructObject2Object, adapted from InstructNeRF2NeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler


@dataclass
class InstructObject2ObjectModelConfig(NerfactoModelConfig):
    """Configuration for the InstructNeRF2NeRFModel."""
    _target: Type = field(default_factory=lambda: InstructObject2ObjectModel)
    use_lpips: bool = True
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1
    """Multiplier for LPIPS loss."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""


class InstructObject2ObjectModel(NerfactoModel): 
    # overwrite all NeRFModel, how to render and edit? 
    # idea: script to render object (transition matrix) and scene and combine images
    # where is the function to modify key of outputs?
    """Model for InstructNeRF2NeRF."""

    config: InstructObject2ObjectModelConfig

    def populate_modules(self): 
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Field for scene
        self.field_scene = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        # self.camera_optimizer_scene: CameraOptimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.num_train_data, device="cpu"
        # )

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )
        
        self.proposal_sampler_scene = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=UniformSampler() # for scene
        )

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters()) + list(self.field_scene.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups
    

    def get_outputs(self, ray_bundle: RayBundle): # how to influence?
        outputs = {}
        outputs = super().get_outputs(ray_bundle)

        ray_samples_scene: RaySamples
        ray_samples_scene, weights_list_scene, ray_samples_list_scene = self.proposal_sampler_scene(ray_bundle, density_fns=self.density_fns)
        scene_outputs = self.field_scene.forward(ray_samples_scene, compute_normals=self.config.predict_normals)

        weights_scene = ray_samples_scene.get_weights(scene_outputs[FieldHeadNames.DENSITY])
        weights_list_scene.append(weights_scene)
        ray_samples_list_scene.append(ray_samples_scene)

        rgb_scene = self.renderer_rgb(rgb=scene_outputs[FieldHeadNames.RGB], weights=weights_scene)

        accumulation_scene = self.renderer_accumulation(weights=weights_scene)

        outputs_scene = {
            "rgb_scene": rgb_scene,
            "accumulation_scene": accumulation_scene,
        }

        outputs.update(outputs_scene)

        if self.training:
            outputs["weights_list_scene"] = weights_list_scene
            outputs["ray_samples_list_scene"] = ray_samples_list_scene

        return outputs
    

    def get_loss_dict(self, outputs, batch, metrics_dict=None): 

        '''
        to find and change : outputs key, batch key
        '''
        loss_dict = {}
        image_object = batch["image_object"].to(self.device)
        image_scene = batch["image_scene"].to(self.device) # inpainted scene

        loss_dict["rgb_loss"] = self.rgb_loss(image_object, outputs["rgb"]) + self.rgb_loss(image_scene, outputs["rgb_scene"])
        # print(self.rgb_loss(image_object, outputs["rgb"]), self.rgb_loss(image_scene, outputs["rgb_scene"]))

        if self.config.use_lpips:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image_object.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            out_patches_scene = (outputs["rgb_scene"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches_scene = (image_scene.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)

            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * (self.lpips(out_patches, gt_patches) + self.lpips(out_patches_scene, gt_patches_scene))

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * (interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]) + interlevel_loss(
                outputs["weights_list_scene"], outputs["ray_samples_list_scene"]))
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        return loss_dict


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image_object"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]

        gt_rgb_scene = batch["image_scene"].to(self.device)  # RGB or RGBA image
        gt_rgb_scene = self.renderer_rgb.blend_background(gt_rgb_scene)  # Blend if RGBA
        predicted_rgb_scene = outputs["rgb_scene"]
        metrics_dict["psnr"] = (self.psnr(predicted_rgb, gt_rgb) + self.psnr(predicted_rgb_scene, gt_rgb_scene))/2

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"]) 
            + distortion_loss(outputs["weights_list_scene"], outputs["ray_samples_list_scene"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
