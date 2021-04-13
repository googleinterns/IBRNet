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

# some code in this file is adapted from https://github.com/augmentedperception/spaces_dataset/

import sys
sys.path.append('../')
import os
import numpy as np
from PIL import Image
import imageio
import torch
from torch.utils.data import Dataset
from .data_utils import quaternion_about_axis, quaternion_matrix, random_crop, random_flip
import json


def view_obj2camera_rgb(view):
    image_path = view.image_path
    intrinsics = view.camera.intrinsics
    h_in_view, w_in_view = view.shape
    rgb = imageio.imread(image_path).astype(np.float32) / 255.
    h_img, w_img = rgb.shape[:2]
    if h_in_view != h_img or w_in_view != w_img:
        intrinsics[0] *= w_img / w_in_view
        intrinsics[1] *= h_img / h_in_view
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w = view.camera.w_f_c
    ref_camera = np.concatenate([list(rgb.shape[:2]), intrinsics_4x4.flatten(), c2w.flatten()])
    return ref_camera, rgb


def view_obj2camera_rgb_path(view):
    img_size = view.shape
    image_path = view.image_path
    intrinsics = view.camera.intrinsics
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w = view.camera.w_f_c
    return image_path, img_size, intrinsics_4x4, c2w


def sample_target_view_for_training(views, input_rig_id, input_ids):
    input_rig_views = views[input_rig_id]
    input_cam_positions = np.array([input_rig_views[i].camera.w_f_c[:3, 3] for i in input_ids])

    remaining_rig_ids = []
    remaining_cam_ids = []

    for i, rig in enumerate(views):
        for j, cam in enumerate(rig):
            if i == input_rig_id and j in input_ids:
                continue
            else:
                cam_loc = views[i][j].camera.w_f_c[:3, 3]
                # if i != input_rig_id:
                #     print(np.min(np.linalg.norm(input_cam_positions - cam_loc, axis=1)))
                if np.min(np.linalg.norm(input_cam_positions - cam_loc, axis=1)) < 0.15:
                    remaining_rig_ids.append(i)
                    remaining_cam_ids.append(j)

    selected_id = np.random.choice(len(remaining_rig_ids))
    selected_view = views[remaining_rig_ids[selected_id]][remaining_cam_ids[selected_id]]
    return selected_view


def get_all_views_in_scene(all_views):
    cameras = []
    rgbs = []
    for rig in all_views:
        for i in range(len(rig)):
            camera, rgb = view_obj2camera_rgb(rig[i])
            cameras.append(camera)
            rgbs.append(rgb)
    return cameras, rgbs


def get_all_views_in_scene_cam_path(all_views):
    c2w_mats = []
    intrinsicss = []
    rgb_paths = []
    img_sizes = []
    for rig in all_views:
        for i in range(len(rig)):
            image_path, img_size, intrinsics_4x4, c2w = view_obj2camera_rgb_path(rig[i])
            rgb_paths.append(image_path)
            intrinsicss.append(intrinsics_4x4)
            c2w_mats.append(c2w)
            img_sizes.append(img_size)
    return rgb_paths, img_sizes, intrinsicss, c2w_mats


def sort_nearby_views_by_angle(query_pose, ref_poses):
    query_direction = np.sum(query_pose[:3, 2:4], axis=-1)
    query_direction = query_direction / np.linalg.norm(query_direction)
    ref_directions = np.sum(ref_poses[:, :3, 2:4], axis=-1)
    ref_directions = ref_directions / np.linalg.norm(ref_directions, axis=-1, keepdims=True)
    inner_product = np.sum(ref_directions * query_direction[None, ...], axis=1)
    sorted_inds = np.argsort(inner_product)[::-1]
    return sorted_inds


class Camera(object):
    """Represents a Camera with intrinsics and world from/to camera transforms.
    Attributes:
      w_f_c: The world from camera 4x4 matrix.
      c_f_w: The camera from world 4x4 matrix.
      intrinsics: The camera intrinsics as a 3x3 matrix.
      inv_intrinsics: The inverse of camera intrinsics matrix.
    """

    def __init__(self, intrinsics, w_f_c):
        """Constructor.
        Args:
          intrinsics: A numpy 3x3 array representing intrinsics.
          w_f_c: A numpy 4x4 array representing wFc.
        """
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        self.w_f_c = w_f_c
        self.c_f_w = np.linalg.inv(w_f_c)


class View(object):
    """Represents an image and associated camera geometry.
    Attributes:
      camera: The camera for this view.
      image: The np array containing the image data.
      image_path: The file path to the image.
      shape: The 2D shape of the image.
    """

    def __init__(self, image_path, shape, camera):
        self.image_path = image_path
        self.shape = shape
        self.camera = camera
        self.image = None


def _WorldFromCameraFromViewDict(view_json):
    """Fills the world from camera transform from the view_json.
    Args:
        view_json: A dictionary of view parameters.
    Returns:
        A 4x4 transform matrix representing the world from camera transform.
    """
    transform = np.identity(4)
    position = view_json['position']
    transform[0:3, 3] = (position[0], position[1], position[2])
    orientation = view_json['orientation']
    angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
    angle = np.linalg.norm(angle_axis)
    epsilon = 1e-7
    if abs(angle) < epsilon:
        # No rotation
        return transform

    axis = angle_axis / angle
    rot_mat = quaternion_matrix(quaternion_about_axis(-angle, axis))
    transform[0:3, 0:3] = rot_mat[0:3, 0:3]
    return transform


def _IntrinsicsFromViewDict(view_params):
    """Fills the intrinsics matrix from view_params.
    Args:
        view_params: Dict view parameters.
    Returns:
        A 3x3 matrix representing the camera intrinsics.
    """
    intrinsics = np.identity(3)
    intrinsics[0, 0] = view_params['focal_length']
    intrinsics[1, 1] = (view_params['focal_length'] * view_params['pixel_aspect_ratio'])
    intrinsics[0, 2] = view_params['principal_point'][0]
    intrinsics[1, 2] = view_params['principal_point'][1]
    return intrinsics


def ReadView(base_dir, view_json):
    return View(
        image_path=os.path.join(base_dir, view_json['relative_path']),
        shape=(int(view_json['height']), int(view_json['width'])),
        camera=Camera(
            _IntrinsicsFromViewDict(view_json),
            _WorldFromCameraFromViewDict(view_json)))


def ReadScene(base_dir):
    """Reads a scene from the directory base_dir."""
    with open(os.path.join(base_dir, 'models.json')) as f:
        model_json = json.load(f)

    all_views = []
    for views in model_json:
        all_views.append([ReadView(base_dir, view_json) for view_json in views])
    return all_views


def InterpolateDepths(near_depth, far_depth, num_depths):
    """Returns num_depths from (far_depth, near_depth), interpolated in inv depth.
    Args:
        near_depth: The first depth.
        far_depth: The last depth.
        num_depths: The total number of depths to create, include near_depth and
        far_depth are always included and other depths are interpolated between
        them, in inverse depth space.
    Returns:
        The depths sorted in descending order (so furthest first). This order is
        useful for back to front compositing.
  """

    inv_near_depth = 1.0 / near_depth
    inv_far_depth = 1.0 / far_depth
    depths = []
    for i in range(0, num_depths):
        fraction = float(i) / float(num_depths - 1)
        inv_depth = inv_far_depth + (inv_near_depth - inv_far_depth) * fraction
        depths.append(1.0 / inv_depth)
    return depths


def ReadViewImages(views):
    """Reads the images for the passed views."""
    for view in views:
        # Keep images unnormalized as uint8 to save RAM and transmission time to
        # and from the GPU.
        view.image = np.array(Image.open(view.image_path))


def WriteNpToImage(np_image, path):
    """Writes an image as a numpy array to the passed path.
        If the input has more than four channels only the first four will be
        written. If the input has a single channel it will be duplicated and
        written as a three channel image.
    Args:
        np_image: A numpy array.
        path: The path to write to.
    Raises:
        IOError: if the image format isn't recognized.
    """

    min_value = np.amin(np_image)
    max_value = np.amax(np_image)
    if min_value < 0.0 or max_value > 255.1:
        print('Warning: Outside image bounds, min: %f, max:%f, clipping.', min_value, max_value)
        np.clip(np_image, 0.0, 255.0)
    if np_image.shape[2] == 1:
        np_image = np.concatenate((np_image, np_image, np_image), axis=2)

    if np_image.shape[2] == 3:
        image = Image.fromarray(np_image.astype(np.uint8))
    elif np_image.shape[2] == 4:
        image = Image.fromarray(np_image.astype(np.uint8), 'RGBA')

    _, ext = os.path.splitext(path)
    ext = ext[1:]
    if ext.lower() == 'png':
        image.save(path, format='PNG')
    elif ext.lower() in ('jpg', 'jpeg'):
        image.save(path, format='JPEG')
    else:
        raise IOError('Unrecognized format for %s' % path)


# only for training
class SpacesDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/spaces_dataset/data/800/')
        self.num_source_views = args.num_source_views
        self.mode = mode
        assert mode in ['train', 'test', 'validation']
        eval_scene_ids = [0, 9, 10, 23, 24, 52, 56, 62, 63, 73]
        train_scene_ids = [i for i in np.arange(0, 100) if i not in eval_scene_ids]
        if mode == 'train':
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in train_scene_ids]
        else:
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in eval_scene_ids]

        self.all_views_scenes = []
        for scene_dir in self.scene_dirs:
            views = ReadScene(scene_dir)
            self.all_views_scenes.append(views)

        self.input_view_types = ["small_quad", "medium_quad", "large_quad", "dense"]
        self.eval_view_indices_dict = {
            "small_quad": [5, 6, 7],
            "medium_quad": [2, 4, 5, 6, 7, 11],
            "large_quad": [1, 2, 4, 5, 6, 7, 8, 10, 11],
            "dense": [5, 7, 10, 11]
        }
        self.input_indices_dict = {
            "small_quad": [1, 2, 10, 11],
            "medium_quad": [1, 3, 10, 12],
            "large_quad": [0, 3, 9, 12],
            "dense": [0, 1, 2, 3, 4, 6, 8, 9, 12, 13, 14, 15]
        }

    def __len__(self):
        return len(self.all_views_scenes)

    def __getitem__(self, idx):
        all_views = self.all_views_scenes[idx]
        num_rigs = len(all_views)
        selected_rig_id = np.random.randint(low=0, high=num_rigs)  # select a rig position
        rig_selected = all_views[selected_rig_id]
        type = np.random.choice(self.input_view_types)  # select an input type
        input_ids = self.input_indices_dict[type]
        if len(input_ids) > self.num_source_views:
            input_ids = np.random.choice(input_ids, self.num_source_views, replace=False)

        ref_cameras = []
        ref_rgbs = []
        w_max, h_max = 0, 0
        for id in input_ids:
            ref_camera, ref_rgb = view_obj2camera_rgb(rig_selected[id])
            ref_rgbs.append(ref_rgb)
            ref_cameras.append(ref_camera)
            h, w = ref_rgb.shape[:2]
            w_max = max(w, w_max)
            h_max = max(h, h_max)

        ref_rgbs_np = np.zeros((len(ref_rgbs), h_max, w_max, 3), dtype=np.float32)
        for i, ref_rgb in enumerate(ref_rgbs):
            orig_h, orig_w = ref_rgb.shape[:2]
            h_start = int((h_max - orig_h) / 2.)
            w_start = int((w_max - orig_w) / 2.)
            ref_rgbs_np[i, h_start:h_start+orig_h, w_start:w_start+orig_w] = ref_rgb
            ref_cameras[i][4] += (w_max - orig_w) / 2.
            ref_cameras[i][8] += (h_max - orig_h) / 2.
            ref_cameras[i][0] = h_max
            ref_cameras[i][1] = w_max

        # select target view
        if self.mode != 'train':
            target_id = np.random.choice(self.eval_view_indices_dict[type])
            target_view = rig_selected[target_id]
            target_camera, target_rgb = view_obj2camera_rgb(target_view)
        else:
            target_view = sample_target_view_for_training(all_views, selected_rig_id, input_ids)
            target_camera, target_rgb = view_obj2camera_rgb(target_view)

        ref_cameras = np.array(ref_cameras)
        if np.random.choice([0, 1], p=[0.5, 0.5]) and self.mode == 'train':
            target_rgb, target_camera, ref_rgbs_np, ref_cameras = random_flip(target_rgb, target_camera,
                                                                              ref_rgbs_np, ref_cameras)

        near_depth = 1.
        far_depth = 100.
        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(target_rgb).float(),
                'camera': torch.from_numpy(target_camera).float(),
                'rgb_path': target_view.image_path,
                'src_rgbs': torch.from_numpy(ref_rgbs_np).float(),
                'src_cameras': torch.from_numpy(np.stack(ref_cameras, axis=0)).float(),
                'depth_range': depth_range
                }


class SpacesFreeDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/spaces_dataset/data/800/')
        self.mode = mode
        self.num_source_views = args.num_source_views
        self.random_crop = True
        assert mode in ['train', 'test', 'validation']
        # eval_scene_ids = [0, 9, 10, 23, 24, 52, 56, 62, 63, 73]
        eval_scene_ids = []
        # use all 100 scenes in spaces dataset for training
        train_scene_ids = [i for i in np.arange(0, 100) if i not in eval_scene_ids]
        if mode == 'train':
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in train_scene_ids]
        else:
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in eval_scene_ids]

        self.all_views_scenes = []
        self.all_rgb_paths_scenes = []
        self.all_intrinsics_scenes = []
        self.all_img_sizes_scenes = []
        self.all_c2w_scenes = []
        for scene_dir in self.scene_dirs:
            views = ReadScene(scene_dir)
            self.all_views_scenes.append(views)
            rgb_paths, img_sizes, intrinsicss, c2w_mats = get_all_views_in_scene_cam_path(views)
            self.all_rgb_paths_scenes.append(rgb_paths)
            self.all_img_sizes_scenes.append(img_sizes)
            self.all_intrinsics_scenes.append(intrinsicss)
            self.all_c2w_scenes.append(c2w_mats)

    def __len__(self):
        return len(self.all_views_scenes)

    def __getitem__(self, idx):
        all_views = self.all_views_scenes[idx]
        num_rigs = len(all_views)
        selected_rig_id = np.random.randint(low=0, high=num_rigs)  # select a rig position
        rig_selected = all_views[selected_rig_id]
        cam_id_selected = np.random.choice(16)
        cam_selected = rig_selected[cam_id_selected]
        render_camera, render_rgb = view_obj2camera_rgb(cam_selected)
        all_c2w_mats = self.all_c2w_scenes[idx]
        all_rgb_paths = self.all_rgb_paths_scenes[idx]
        all_intrinsics = self.all_intrinsics_scenes[idx]
        all_img_sizes = self.all_img_sizes_scenes[idx]
        sorted_ids = sort_nearby_views_by_angle(render_camera[-16:].reshape(4, 4), np.array(all_c2w_mats))
        nearby_view_ids_selected = np.random.choice(sorted_ids[1:],
                                                    self.num_source_views, replace=False)

        ref_cameras = []
        ref_rgbs = []
        w_max, h_max = 0, 0
        for id in nearby_view_ids_selected:
            rgb_path = all_rgb_paths[id]
            ref_rgb = imageio.imread(rgb_path).astype(np.float32) / 255.
            h_in_view, w_in_view = all_img_sizes[id]
            h_img, w_img = ref_rgb.shape[:2]
            ref_rgbs.append(ref_rgb)
            ref_intrinsics = all_intrinsics[id]
            if h_in_view != h_img or w_in_view != w_img:
                ref_intrinsics[0] *= w_img / w_in_view
                ref_intrinsics[1] *= h_img / h_in_view
            ref_c2w = all_c2w_mats[id]
            ref_camera = np.concatenate([list(ref_rgb.shape[:2]), ref_intrinsics.flatten(), ref_c2w.flatten()])
            ref_cameras.append(ref_camera)
            h, w = ref_rgb.shape[:2]
            w_max = max(w, w_max)
            h_max = max(h, h_max)

        ref_rgbs_np = np.ones((len(ref_rgbs), h_max, w_max, 3), dtype=np.float32)
        for i, ref_rgb in enumerate(ref_rgbs):
            orig_h, orig_w = ref_rgb.shape[:2]
            h_start = int((h_max - orig_h) / 2.)
            w_start = int((w_max - orig_w) / 2.)
            ref_rgbs_np[i, h_start:h_start+orig_h, w_start:w_start+orig_w] = ref_rgb
            ref_cameras[i][4] += (w_max - orig_w) / 2.
            ref_cameras[i][8] += (h_max - orig_h) / 2.
            ref_cameras[i][0] = h_max
            ref_cameras[i][1] = w_max

        ref_cameras = np.array(ref_cameras)
        if self.mode == 'train' and self.random_crop:
            render_rgb, render_camera, ref_rgbs_np, ref_cameras = random_crop(render_rgb, render_camera,
                                                                              ref_rgbs_np, ref_cameras)

        if self.mode == 'train' and np.random.choice([0, 1]):
            render_rgb, render_camera, ref_rgbs_np, ref_cameras = random_flip(render_rgb, render_camera,
                                                                              ref_rgbs_np, ref_cameras)

        near_depth = 0.7
        far_depth = 100
        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(render_rgb).float(),
                'camera': torch.from_numpy(render_camera).float(),
                'rgb_path': cam_selected.image_path,
                'src_rgbs': torch.from_numpy(ref_rgbs_np).float(),
                'src_cameras': torch.from_numpy(np.stack(ref_cameras, axis=0)).float(),
                'depth_range': depth_range
                }

