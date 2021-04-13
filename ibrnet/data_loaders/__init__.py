# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .google_scanned_objects import *
from .realestate import *
from .deepvoxels import *
from .realestate import *
from .llff import *
from .llff_test import *
from .ibrnet_collected import *
from .realestate import *
from .spaces_dataset import *
from .nerf_synthetic import *

dataset_dict = {
    'spaces': SpacesFreeDataset,
    'google_scanned': GoogleScannedDataset,
    'realestate': RealEstateDataset,
    'deepvoxels': DeepVoxelsDataset,
    'nerf_synthetic': NerfSyntheticDataset,
    'llff': LLFFDataset,
    'ibrnet_collected': IBRNetCollectedDataset,
    'llff_test': LLFFTestDataset
}