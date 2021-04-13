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


import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


# only for training
class GoogleScannedDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/google_scanned_objects/')
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, '*'))

        all_rgb_files = []
        all_pose_files = []
        all_intrinsics_files = []
        num_files = 250
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, 'rgb', f)
                         for f in sorted(os.listdir(os.path.join(scene_path, 'rgb')))]
            pose_files = [f.replace('rgb', 'pose').replace('png', 'txt') for f in rgb_files]
            intrinsics_files = [f.replace('rgb', 'intrinsics').replace('png', 'txt') for f in rgb_files]

            if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) \
                    < num_files:
                print(scene_path)
                continue

            all_rgb_files.append(rgb_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        id_render = np.random.choice(np.arange(len(rgb_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]
        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(render_pose,
                                            train_poses,
                                            self.num_source_views*subsample_factor,
                                            tar_id=id_render,
                                            angular_dist_method='vector')
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        assert id_render not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = id_render

        rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.

        intrinsics = np.loadtxt(intrinsics_files[id_render])
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics, render_pose.flatten())).astype(np.float32)

        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        depth_range = torch.tensor([near_depth, far_depth])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.
            pose = np.loadtxt(pose_files[id])
            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id])
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics, pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {'rgb': torch.from_numpy(rgb),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_files[id_render],
                'src_rgbs': torch.from_numpy(src_rgbs),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range
                }

