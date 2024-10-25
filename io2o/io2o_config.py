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
Instruct-Object2Object configuration file.
Adapted from Instruct-NeRF2NeRF
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from io2o.io2o_datamanager import InstructObject2ObjectDataManagerConfig
from io2o.io2o import InstructObject2ObjectModelConfig
from io2o.io2o_pipeline import InstructObject2ObjectPipelineConfig
from io2o.io2o_trainer import InstructObject2ObjectTrainerConfig

io2o_method = MethodSpecification(
    config=InstructObject2ObjectTrainerConfig(
        method_name="io2o",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=25000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=InstructObject2ObjectPipelineConfig(
            datamanager=InstructObject2ObjectDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=1024,
                patch_size=32,
            ),
            model=InstructObject2ObjectModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
            ip2p_use_full_precision=False
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-Object2Object method (adapted from Instruct-Nerf2Nerf): specify object and scene to edit",
)
