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
import sys
import json
sys.path.append('../')
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta['camera_angle_x'])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta['frames'][0]['file_path'] + '.png'))
    H, W = img.shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta['frames']):
        rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'][2:] + '.png')
        rgb_files.append(rgb_file)
        c2w = np.array(frame['transform_matrix'])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics]*len(meta['frames'])), c2w_mats


def get_intrinsics_from_hwf(h, w, focal):
    return np.array([[focal, 0, 1.0*w/2, 0],
                     [0, focal, 1.0*h/2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


class NerfSyntheticDataset(Dataset):
    def __init__(self, args, mode,
                 # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
                 scenes=(), **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/nerf_synthetic/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        all_scenes = ('chair', 'drums', 'lego', 'hotdog', 'materials', 'mic', 'ship')
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_file = os.path.join(self.scene_path, 'transforms_{}.json'.format(mode))
            rgb_files, intrinsics, poses = read_cameras(pose_file)
            if self.mode != 'train':
                rgb_files = rgb_files[::self.testskip]
                intrinsics = intrinsics[::self.testskip]
                poses = poses[::self.testskip]
            self.render_rgb_files.extend(rgb_files)
            self.render_poses.extend(poses)
            self.render_intrinsics.extend(intrinsics)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        train_pose_file = os.path.join('/'.join(rgb_file.split('/')[:-2]), 'transforms_train.json')
        train_rgb_files, train_intrinsics, train_poses = read_cameras(train_pose_file)

        if self.mode == 'train':
            id_render = int(os.path.basename(rgb_file)[:-4].split('_')[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), render_intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                int(self.num_source_views*subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        near_depth = 2.
        far_depth = 6.

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }

