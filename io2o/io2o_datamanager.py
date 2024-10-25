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
Instruct-Object2Object Datamanager.
"""

import torch

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console
import subprocess
import cv2
import os

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
from typing import (
    Dict,
    Tuple,
    Type,
)

from typing_extensions import TypeVar

from nerfstudio.cameras.rays import RayBundle

from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datasets.base_dataset import InputDataset
from PIL import Image
import numpy as np

TEXTURE_PATH = "/workspace/nerfstudio/instruct-object2object/io2o/texture-synthesis"

def save_image_tensor(input_tensor: torch.Tensor, filename):
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = (input_tensor.numpy()*255).astype(np.uint8)
    
    im = Image.fromarray(input_tensor)
    im.save(filename)

'''
Overwrite this to include masks
'''
class io2oDataset(InputDataset):
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["mask"]

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        data = super().get_data(image_idx)
        assert data["mask"] is not None

        def inpaint(img, mask, image_idx):
            mask = mask.numpy()*255
            img *= 255
            img = img.numpy()
            ma = np.where(mask < 10, 0, 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(ma, kernel, iterations=8)
            original_path = os.getcwd()
            os.chdir(TEXTURE_PATH)
            print(os.getcwd())
            cv2.imwrite("{}_image.jpg".format(image_idx), (img))
            size = (img.shape[1], img.shape[0])
            cv2.imwrite("{}_mask.jpg".format(image_idx), mask)
            command = 'cargo run --release -- --out-size 1080 --inpaint temp_mask.jpg -o {}.jpg generate temp_image.jpg'.format(image_idx)
            # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            os.system(command)
            inpainted = cv2.resize(cv2.imread("{}.jpg".format(image_idx)), size)
            os.chdir(original_path)
            return torch.tensor(inpainted/255).to(torch.device('cuda:0'))

        # split object out and inpainting
        mask = data["mask"].clone().cpu()
        mask = torch.squeeze(mask)
        image_o = data["image"].clone()
        image_s = data["image"].clone()
        image_o[mask == True] = 0.05
        image_s[mask == False] = 0.7

        del data["mask"]
        del data["image"]
        data["image_object"] = image_o
        data["image_scene"] = inpaint(image_s, mask, image_idx)
        data["image"] = data["image_object"]

        return data

CONSOLE = Console(width=120)
TDataset = TypeVar("TDataset", bound=io2oDataset, default=io2oDataset)

@dataclass
class InstructObject2ObjectDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the InstructObject2ObjectDataManager."""

    _target: Type = field(default_factory=lambda: InstructObject2ObjectDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

class InstructObject2ObjectDataManager(VanillaDataManager):
    """Data manager for InstructObject2Object."""

    '''
    split object out and change the keys of train set
    '''
    config: InstructObject2ObjectDataManagerConfig

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        return io2oDataset
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers= 1, #self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device),)

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch['image_object'] = self.image_batch['image_object'].clone().to("cpu")
        self.original_image_batch['image_scene'] = self.image_batch['image_scene'].clone().to("cpu")
        self.original_image_batch['image_idx'] = self.image_batch['image_idx'].clone().to("cpu")


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

