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


'''
This file verifies if the camera poses are correct by looking at the epipolar geometry.
Given a pair of images and their relative pose, we sample a bunch of discriminative points in the first image,
we draw its corresponding epipolar line in the other image. If the camera pose is correct,
the epipolar line should pass the ground truth correspondence location.
'''


import cv2
import numpy as np


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_view_geometry(intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    '''
    :param intrinsics1: 4 by 4 matrix
    :param extrinsics1: 4 by 4 W2C matrix
    :param intrinsics2: 4 by 4 matrix
    :param extrinsics2: 4 by 4 W2C matrix
    :return:
    '''
    relative_pose = extrinsics2.dot(np.linalg.inv(extrinsics1))
    R = relative_pose[:3, :3]
    T = relative_pose[:3, 3]
    tx = skew(T)
    E = np.dot(tx, R)
    F = np.linalg.inv(intrinsics2[:3, :3]).T.dot(E).dot(np.linalg.inv(intrinsics1[:3, :3]))

    return E, F, relative_pose


def drawpointslines(img1, img2, lines1, pts2, color):
    '''
    draw corresponding epilines on img1 for the points in img2
    '''

    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt2, cl in zip(lines1, pts2, color):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cl = tuple(cl.tolist())
        img1 = cv2.line(img1, (x0,y0), (x1,y1), cl, 1)
        img2 = cv2.circle(img2, tuple(pt2), 5, cl, -1)
    return img1, img2


def epipolar(coord1, F, img1, img2):
    # compute epipole
    pts1 = coord1.astype(int).T
    color = np.random.randint(0, high=255, size=(len(pts1), 3))
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    if lines2 is None:
        return None
    lines2 = lines2.reshape(-1,3)

    img3, img4 = drawpointslines(img2,img1,lines2,pts1,color)
    ## print(img3.shape)
    ## print(np.concatenate((img4, img3)).shape)
    ## cv2.imwrite('vis.png', np.concatenate((img4, img3), axis=1))
    h_max = max(img3.shape[0], img4.shape[0])
    w_max = max(img3.shape[1], img4.shape[1])
    out = np.ones((h_max, w_max*2, 3))
    out[:img4.shape[0], :img4.shape[1], :] = img4
    out[:img3.shape[0], w_max:w_max+img3.shape[1], :] = img3

    # return np.concatenate((img4, img3), axis=1)
    return out


def verify_data(img1, img2, intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    E, F, relative_pose = two_view_geometry(intrinsics1, extrinsics1,
                                            intrinsics2, extrinsics2)

    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
    # kp1 = sift.detect(img1, mask=None)
    # coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1]).T

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1[:20]]).T
    return epipolar(coord1, F, img1, img2)


def calc_angles(c2w_1, c2w_2):
    c1 = c2w_1[:3, 3:4]
    c2 = c2w_2[:3, 3:4]

    c1 = c1 / np.linalg.norm(c1)
    c2 = c2 / np.linalg.norm(c2)
    return np.rad2deg(np.arccos(np.dot(c1.T, c2)))


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    import os
    from ibrnet.data_loaders.google_scanned_objects import GoogleScannedDataset
    from config import config_parser

    parser = config_parser()
    args = parser.parse_args()
    dataset = GoogleScannedDataset(args, mode='train')
    out_dir = 'data_verify'
    print('saving output to {}...'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    for k, data in enumerate(dataset):
        rgb = data['rgb'].cpu().numpy()
        camera = data['camera'].cpu().numpy()
        src_rgbs = data['src_rgbs'].cpu().numpy()
        src_cameras = data['src_cameras'].cpu().numpy()
        i = np.random.randint(low=0, high=len(src_rgbs))
        rgb_i = src_rgbs[i]
        cameras_i = src_cameras[i]
        intrinsics1 = camera[2:18].reshape(4, 4)
        intrinsics2 = cameras_i[2:18].reshape(4, 4)
        extrinsics1 = np.linalg.inv(camera[-16:].reshape(4, 4))
        extrinsics2 = np.linalg.inv(cameras_i[-16:].reshape(4, 4))

        im = verify_data(np.uint8(rgb*255.), np.uint8(rgb_i*255.),
                         intrinsics1, extrinsics1,
                         intrinsics2, extrinsics2)
        if im is not None:
            cv2.imwrite(os.path.join(out_dir, '{:03d}.png'.format(k)), im)


