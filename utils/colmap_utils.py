import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

# from pycolmap import SceneManager
import array
import numpy as np
import os
import struct
from collections import OrderedDict
from itertools import combinations
from pycolmap.camera import Camera
from pycolmap.image import Image
from pycolmap.rotation import Quaternion

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.cam_nums = 0
        
        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "colmap/sparse/0")
            if not os.path.exists(colmap_dir):
                if all(os.path.exists(os.path.join(data_dir, f)) for f in ["cameras.txt", "images.txt", "points3D.txt"]):
                    colmap_dir = data_dir
                    images_dir = os.path.join(data_dir, "images")
                else:
                    colmap_dir = os.path.join(data_dir, "sparse")
                    if not os.path.exists(colmap_dir) or not all(os.path.exists(os.path.join(colmap_dir, f)) for f in ["cameras.txt", "images.txt", "points3D.txt"]):
                        colmap_dir = None

        assert colmap_dir is not None and os.path.exists(colmap_dir), f"COLMAP directory with required files not found in {data_dir}"

        print(f"[LOG] Colmap read path is {colmap_dir}...")
        manager = SceneManager(colmap_dir, image_path=images_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        processed_count = 0
        print(f"[Parser] Loading {len(imdata)} images...")
        
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        
            processed_count += 1
            progress = processed_count / len(imdata)
            print(f"\r[Parser] Loading images: {processed_count}/{len(imdata)} ({progress:.1%})", end='', flush=True)

        print()  
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        self.cam_nums = len(camera_ids)
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        if "3dv-dataset-nerfstudio" in data_dir:
            colmap_files = sorted(_get_rel_paths(colmap_image_dir), key=lambda x: int(x.split(".")[0].split("_")[-1]))
            image_files = sorted(_get_rel_paths(image_dir), key=lambda x: int(x.split(".")[0].split("_")[-1]))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_names = colmap_files
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        elif "DL3DV-Benchmark" in data_dir:
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            image_files = sorted(_get_rel_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            if len(colmap_files) != len(image_names):
                print(f"Warning: colmap_files: {len(colmap_files)}, image_names: {len(image_names)}")
                image_names = colmap_files
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        else:
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            image_files = sorted(_get_rel_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, os.path.basename(f)) for f in image_names]
            # image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.alpha_mask_paths = None  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (actual_width, actual_height)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        
        indices = np.arange(len(self.parser.image_names))
        if self.parser.test_every == 1:
            image_names = sorted(_get_rel_paths(f"{self.parser.data_dir}/images"), key=lambda x: int(x.split(".")[0].split("_")[-1]))
            assert len(image_names) == len(self.parser.image_names)
            if split == "train":
                self.indices = [ind for ind in indices if "_train_" in image_names[ind]]
            else:
                self.indices = [ind for ind in indices if "_eval_" in image_names[ind]]
        elif self.parser.test_every == 0:
            self.indices = indices
        else:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every == 0]
            else:
                self.indices = indices[indices % self.parser.test_every != 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()
        
        if self.parser.alpha_mask_paths is not None:
            alpha_mask = imageio.imread(self.parser.alpha_mask_paths[index], mode="L")[:,:,None] / 255.0
            if self.patch_size is not None:
                alpha_mask = alpha_mask[y : y + self.patch_size, x : x + self.patch_size]
            data["alpha_mask"] = torch.from_numpy(alpha_mask).float()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


# Fixed SceneManager for Python 3 compatibility
# Based on pycolmap SceneManager with minimal changes



class SceneManager:
    INVALID_POINT3D = np.uint64(-1)

    def __init__(self, colmap_results_folder, image_path=None):
        self.folder = colmap_results_folder
        if not self.folder.endswith('/'):
            self.folder += '/'

        self.image_path = None
        self.load_colmap_project_file(image_path=image_path)

        self.cameras = OrderedDict()
        self.images = OrderedDict()
        self.name_to_image_id = dict()

        self.last_camera_id = 0
        self.last_image_id = 0

        # Nx3 array of point3D xyz's
        self.points3D = np.zeros((0, 3))

        # for each element in points3D, stores the id of the point
        self.point3D_ids = np.empty(0)

        # point3D_id => index in self.points3D
        self.point3D_id_to_point3D_idx = dict()

        # point3D_id => [(image_id, point2D idx in image)]
        self.point3D_id_to_images = dict()

        self.point3D_colors = np.zeros((0, 3), dtype=np.uint8)
        self.point3D_errors = np.zeros(0)

    def load_colmap_project_file(self, project_file=None, image_path=None):
        if project_file is None:
            project_file = self.folder + 'project.ini'

        self.image_path = image_path

        if self.image_path is None:
            try:
                with open(project_file, 'r') as f:
                    for line in iter(f.readline, ''):
                        if line.startswith('image_path'):
                            self.image_path = line[11:].strip()
                            break
            except:
                pass

        if self.image_path is None:
            print('Warning: image_path not found for reconstruction')
        elif not self.image_path.endswith('/'):
            self.image_path += '/'

    def load(self):
        self.load_cameras()
        self.load_images()
        self.load_points3D()

    def load_cameras(self, input_file=None):
        if input_file is None:
            input_file = self.folder + 'cameras.bin'
            if os.path.exists(input_file):
                self._load_cameras_bin(input_file)
            else:
                input_file = self.folder + 'cameras.txt'
                if os.path.exists(input_file):
                    self._load_cameras_txt(input_file)
                else:
                    raise IOError('no cameras file found')
    
    def _load_cameras_bin(self, input_file):
        self.cameras = OrderedDict()

        with open(input_file, 'rb') as f:
            num_cameras = struct.unpack('L', f.read(8))[0]

            for _ in range(num_cameras):
                camera_id, camera_type, w, h = struct.unpack('IiLL', f.read(24))
                num_params = Camera.GetNumParams(camera_type)
                params = struct.unpack('d' * num_params, f.read(8 * num_params))
                self.cameras[camera_id] = Camera(camera_type, w, h, params)
                self.last_camera_id = max(self.last_camera_id, camera_id)

    def _load_cameras_txt(self, input_file):
        self.cameras = OrderedDict()

        with open(input_file, 'r') as f:
            for line in iter(lambda: f.readline().strip(), ''):
                if not line or line.startswith('#'):
                    continue

                data = line.split()
                camera_id = int(data[0])
                # FIX: wrap map() with list() for Python 3
                self.cameras[camera_id] = Camera(
                    data[1], int(data[2]), int(data[3]), list(map(float, data[4:])))
                self.last_camera_id = max(self.last_camera_id, camera_id)

    def load_images(self, input_file=None):
        if input_file is None:
            input_file = self.folder + 'images.bin'
            if os.path.exists(input_file):
                self._load_images_bin(input_file)
            else:
                input_file = self.folder + 'images.txt'
                if os.path.exists(input_file):
                    self._load_images_txt(input_file)
                else:
                    raise IOError('no images file found')

    def _load_images_bin(self, input_file):
        self.images = OrderedDict()

        with open(input_file, 'rb') as f:
            num_images = struct.unpack('L', f.read(8))[0]
            image_struct = struct.Struct('<I 4d 3d I')
            for _ in range(num_images):
                data = image_struct.unpack(f.read(image_struct.size))
                image_id = data[0]
                q = Quaternion(np.array(data[1:5]))
                t = np.array(data[5:8])
                camera_id = data[8]
                name = b''.join(c for c in iter(lambda: f.read(1), b'\x00')).decode()

                image = Image(name, camera_id, q, t)
                num_points2D = struct.unpack('Q', f.read(8))[0]

                points_array = array.array('d')
                points_array.fromfile(f, 3 * num_points2D)
                points_elements = np.array(points_array).reshape((num_points2D, 3))
                image.points2D = points_elements[:, :2]

                ids_array = array.array('Q')
                ids_array.frombytes(points_elements[:, 2].tobytes())
                image.point3D_ids = np.array(ids_array, dtype=np.uint64).reshape(
                    (num_points2D,))

                self.images[image_id] = image
                self.name_to_image_id[image.name] = image_id
                self.last_image_id = max(self.last_image_id, image_id)

    def _load_images_txt(self, input_file):
        self.images = OrderedDict()

        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                data = line.split()
                if len(data) < 10:
                    continue

                try:
                    image_id = int(data[0])
                    quat_data = list(map(float, data[1:5]))  # QW, QX, QY, QZ
                    trans_data = list(map(float, data[5:8]))  # TX, TY, TZ
                    camera_id = int(data[8])
                    image_name = data[9]

                    image = Image(image_name, camera_id,
                                Quaternion(np.array(quat_data)),
                                np.array(trans_data))

                    self.images[image_id] = image
                    self.name_to_image_id[image.name] = image_id
                    self.last_image_id = max(self.last_image_id, image_id)

                except Exception as e:
                    print(f"[ERROR] Failed to parse image line: {e}")
                    continue

    def load_points3D(self, input_file=None):
        if input_file is None:
            input_file = self.folder + 'points3D.bin'
            if os.path.exists(input_file):
                self._load_points3D_bin(input_file)
            else:
                input_file = self.folder + 'points3D.txt'
                if os.path.exists(input_file):
                    self._load_points3D_txt(input_file)
                else:
                    raise IOError('no points3D file found')

    def _load_points3D_bin(self, input_file):
        with open(input_file, 'rb') as f:
            num_points3D = struct.unpack('L', f.read(8))[0]

            self.points3D = np.empty((num_points3D, 3))
            self.point3D_ids = np.empty(num_points3D, dtype=np.uint64)
            self.point3D_colors = np.empty((num_points3D, 3), dtype=np.uint8)
            self.point3D_id_to_point3D_idx = dict()
            self.point3D_id_to_images = dict()
            self.point3D_errors = np.empty(num_points3D)

            data_struct = struct.Struct('<Q 3d 3B d Q')

            for i in range(num_points3D):
                data = data_struct.unpack(f.read(data_struct.size))
                self.point3D_ids[i] = data[0]
                self.points3D[i] = data[1:4]
                self.point3D_colors[i] = data[4:7]
                self.point3D_errors[i] = data[7]
                track_len = data[8]

                self.point3D_id_to_point3D_idx[self.point3D_ids[i]] = i

                data = struct.unpack(f'{2*track_len}I', f.read(2 * track_len * 4))

                self.point3D_id_to_images[self.point3D_ids[i]] = \
                    np.array(data, dtype=np.uint32).reshape(track_len, 2)

    def _load_points3D_txt(self, input_file):
        self.points3D = []
        self.point3D_ids = []
        self.point3D_colors = []
        self.point3D_id_to_point3D_idx = dict()
        self.point3D_id_to_images = dict()
        self.point3D_errors = []

        with open(input_file, 'r') as f:
            for line in iter(lambda: f.readline().strip(), ''):
                if not line or line.startswith('#'):
                    continue

                data = line.split()
                point3D_id = np.uint64(data[0])

                self.point3D_ids.append(point3D_id)
                self.point3D_id_to_point3D_idx[point3D_id] = len(self.points3D)
                # FIX: wrap map() with list() for Python 3
                self.points3D.append(list(map(np.float64, data[1:4])))
                self.point3D_colors.append(list(map(np.uint8, data[4:7])))
                self.point3D_errors.append(np.float64(data[7]))

                # FIX: wrap map() with list() and handle reshape error
                # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
                try:
                    image_id = int(data[8])
                    point2d_idx = int(data[9])
                    if image_id > 0:
                        self.point3D_id_to_images[point3D_id] = \
                            np.array([[image_id, point2d_idx]], dtype=np.uint32)
                    else:
                        print(f"[Warning] Invalid image_id: {image_id}")
                        self.point3D_id_to_images[point3D_id] = np.empty((0, 2), dtype=np.uint32)
                except:
                    print(f"[Warning] Invalid track data for point3D_id: {point3D_id}")
                    self.point3D_id_to_images[point3D_id] = np.empty((0, 2), dtype=np.uint32)
                                                    
                    
        self.points3D = np.array(self.points3D)
        self.point3D_ids = np.array(self.point3D_ids)
        self.point3D_colors = np.array(self.point3D_colors)
        self.point3D_errors = np.array(self.point3D_errors)

    def get_camera(self, camera_id):
        return self.cameras[camera_id]

    def get_image_from_name(self, image_name):
        image_id = self.name_to_image_id[image_name]
        return image_id, self.images[image_id]
    

    #---------------------------------------------------------------------------

    def save(self, output_folder, binary=True):
        self.save_cameras(output_folder, binary=binary)
        self.save_images(output_folder, binary=binary)
        self.save_points3D(output_folder, binary=binary)

    #---------------------------------------------------------------------------

    def save_cameras(self, output_folder, output_file=None, binary=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = 'cameras.bin' if binary else 'cameras.txt'

        output_file = os.path.join(output_folder, output_file)

        if binary:
            self._save_cameras_bin(output_file)
        else:
            self._save_cameras_txt(output_file)
    
    def _save_cameras_bin(self, output_file):
        with open(output_file, 'wb') as fid:
            fid.write(struct.pack('L', len(self.cameras)))

            camera_struct = struct.Struct('IiLL')

            for camera_id, camera in sorted(self.cameras.iteritems()):
                fid.write(camera_struct.pack(
                    camera_id, camera.camera_type, camera.width, camera.height))
                # TODO (True): should move this into the Camera class
                fid.write(camera.get_params().tobytes())

    def _save_cameras_txt(self, output_file):
        with open(output_file, 'w') as fid:
            print>>fid, '# Camera list with one line of data per camera:'
            print>>fid, '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]'
            print>>fid, '# Number of cameras:', len(self.cameras)

            for camera_id, camera in sorted(self.cameras.iteritems()):
                print>>fid, camera_id, camera

    #---------------------------------------------------------------------------

    def save_images(self, output_folder, output_file=None, binary=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = 'images.bin' if binary else 'images.txt'

        output_file = os.path.join(output_folder, output_file)

        if binary:
            self._save_images_bin(output_file)
        else:
            self._save_images_txt(output_file)

    def _save_images_bin(self, output_file):
        with open(output_file, 'wb') as fid:
            fid.write(struct.pack('L', len(self.images)))

            for image_id, image in self.images.iteritems():
                fid.write(struct.pack('I', image_id))
                fid.write(image.q.q.tobytes())
                fid.write(image.tvec.tobytes())
                fid.write(struct.pack('I', image.camera_id))
                fid.write(image.name + '\0')
                fid.write(struct.pack('L', len(image.points2D)))
                data = np.rec.fromarrays(
                    (image.points2D[:,0], image.points2D[:,1], image.point3D_ids))
                fid.write(data.tobytes())

    def _save_images_txt(self, output_file):
        with open(output_file, 'w') as fid:
            print>>fid, '# Image list with two lines of data per image:'
            print>>fid, '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME'
            print>>fid, '#   POINTS2D[] as (X, Y, POINT3D_ID)'
            print>>fid, '# Number of images: {},'.format(len(self.images)),
            print>>fid, 'mean observations per image: unknown'

            for image_id, image in self.images.iteritems():
                print>>fid, image_id,
                print>>fid, ' '.join(str(qi) for qi in image.q.q),
                print>>fid, ' '.join(str(ti) for ti in image.tvec),
                print>>fid, image.camera_id, image.name

                data = np.rec.fromarrays(
                    (image.points2D[:,0], image.points2D[:,1],
                     image.point3D_ids.astype(np.int64)))
                if len(data) > 0:
                    np.savetxt(fid, data, '%.2f %.2f %d', newline=' ')
                    fid.seek(-1, os.SEEK_CUR)
                fid.write('\n')

    #---------------------------------------------------------------------------

    def save_points3D(self, output_folder, output_file=None, binary=True):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = 'points3D.bin' if binary else 'points3D.txt'

        output_file = os.path.join(output_folder, output_file)

        if binary:
            self._save_points3D_bin(output_file)
        else:
            self._save_points3D_txt(output_file)

    def _save_points3D_bin(self, output_file):
        num_valid_points3D = sum(
            1 for point3D_idx in self.point3D_id_to_point3D_idx.itervalues()
            if point3D_idx != SceneManager.INVALID_POINT3D)

        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        with open(output_file, 'wb') as fid:
            fid.write(struct.pack('L', num_valid_points3D))

            for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
                if point3D_idx == SceneManager.INVALID_POINT3D:
                    continue

                fid.write(struct.pack('L', point3D_id))
                fid.write(self.points3D[point3D_idx].tobytes())
                fid.write(self.point3D_colors[point3D_idx].tobytes())
                fid.write(self.point3D_errors[point3D_idx].tobytes())
                fid.write(
                    struct.pack('L', len(self.point3D_id_to_images[point3D_id])))
                fid.write(self.point3D_id_to_images[point3D_id].tobytes())

    def _save_points3D_txt(self, output_file):
        num_valid_points3D = sum(
            1 for point3D_idx in self.point3D_id_to_point3D_idx.itervalues()
            if point3D_idx != SceneManager.INVALID_POINT3D)

        array_to_string = lambda arr: ' '.join(str(x) for x in arr)

        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        with open(output_file, 'w') as fid:
            print>>fid, '# 3D point list with one line of data per point:'
            print>>fid, '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as ',
            print>>fid, '(IMAGE_ID, POINT2D_IDX)'
            print>>fid, '# Number of points: {},'.format(num_valid_points3D),
            print>>fid, 'mean track length: unknown'

            for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
                if point3D_idx == SceneManager.INVALID_POINT3D:
                    continue

                print>>fid, point3D_id,
                print>>fid, array_to_string(self.points3D[point3D_idx]),
                print>>fid, array_to_string(self.point3D_colors[point3D_idx]),
                print>>fid, self.point3D_errors[point3D_idx],
                print>>fid, array_to_string(
                    self.point3D_id_to_images[point3D_id].flat)

    #---------------------------------------------------------------------------

    # return the image id associated with a given image file
    def get_image_from_name(self, image_name):
        image_id = self.name_to_image_id[image_name]
        return image_id, self.images[image_id]

    #---------------------------------------------------------------------------

    def get_camera(self, camera_id):
        return self.cameras[camera_id]

    #---------------------------------------------------------------------------

    def get_points3D(self, image_id, return_points2D=True, return_colors=False):
        image = self.images[image_id]

        mask = (image.point3D_ids != SceneManager.INVALID_POINT3D)

        point3D_idxs = np.array([
            self.point3D_id_to_point3D_idx[point3D_id]
            for point3D_id in image.point3D_ids[mask]])
        # detect filtered points
        filter_mask = (point3D_idxs != SceneManager.INVALID_POINT3D)
        point3D_idxs = point3D_idxs[filter_mask]
        result = [self.points3D[point3D_idxs,:]]

        if return_points2D:
            mask[mask] &= filter_mask
            result += [image.points2D[mask]]
        if return_colors:
            result += [self.point3D_colors[point3D_idxs,:]]

        return result if len(result) > 1 else result[0]

    #---------------------------------------------------------------------------

    def point3D_valid(self, point3D_id):
        return (self.point3D_id_to_point3D_idx[point3D_id] !=
                SceneManager.INVALID_POINT3D)

    #---------------------------------------------------------------------------

    def get_filtered_points3D(self, return_colors=False):
        point3D_idxs = [
            idx for idx in self.point3D_id_to_point3D_idx.values()
            if idx != SceneManager.INVALID_POINT3D]
        result = [self.points3D[point3D_idxs,:]]
        
        if return_colors:
            result += [self.point3D_colors[point3D_idxs,:]]

        return result if len(result) > 1 else result[0]

    #---------------------------------------------------------------------------

    # return 3D points shared by two images
    def get_shared_points3D(self, image_id1, image_id2):
        point3D_ids = (
                set(self.images[image_id1].point3D_ids) &
                set(self.images[image_id2].point3D_ids))
        point3D_ids.discard(SceneManager.INVALID_POINT3D)

        point3D_idxs = np.array([self.point3D_id_to_point3D_idx[point3D_id]
            for point3D_id in point3D_ids])

        return self.points3D[point3D_idxs,:]

    #---------------------------------------------------------------------------

    # project *all* 3D points into image, return their projection coordinates,
    # as well as their 3D positions
    def get_viewed_points(self, image_id):
        image = self.images[image_id]

        # get unfiltered points
        point3D_idxs = set(self.point3D_id_to_point3D_idx.itervalues())
        point3D_idxs.discard(SceneManager.INVALID_POINT3D)
        point3D_idxs = list(point3D_idxs)
        points3D = self.points3D[point3D_idxs,:]

        # orient points relative to camera
        R = image.q.ToR()
        points3D = points3D.dot(R.T) + image.tvec[np.newaxis,:]
        points3D = points3D[points3D[:,2] > 0,:] # keep points with positive z

        # put points into image coordinates
        camera = self.cameras[image.camera_id]
        points2D = points3D.dot(camera.get_camera_matrix().T)
        points2D = points2D[:,:2] / points2D[:,2][:,np.newaxis]

        # keep points that are within the image
        mask = (
            (points2D[:,0] >= 0) &
            (points2D[:,1] >= 0) &
            (points2D[:,0] < camera.width - 1) &
            (points2D[:,1] < camera.height - 1))

        return points2D[mask,:], points3D[mask,:]

    #---------------------------------------------------------------------------

    def add_camera(self, camera):
        self.last_camera_id += 1
        self.cameras[self.last_camera_id] = camera
        return self.last_camera_id

    #---------------------------------------------------------------------------

    def add_image(self, image):
        self.last_image_id += 1
        self.images[self.last_image_id] = image
        return self.last_image_id

    #---------------------------------------------------------------------------

    def delete_images(self, image_list):
        # delete specified images
        for image_id in image_list:
            if image_id in self.images:
                del self.images[image_id]

        keep_set = set(self.images.iterkeys())

        # delete references to specified images, and ignore any points that are
        # invalidated
        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
            if point3D_idx == SceneManager.INVALID_POINT3D:
                continue

            mask = np.array([
                image_id in keep_set
                for image_id in self.point3D_id_to_images[point3D_id][:,0]])
            if np.any(mask):
                self.point3D_id_to_images[point3D_id] = \
                    self.point3D_id_to_images[point3D_id][mask]
            else:
                self.point3D_id_to_point3D_idx[point3D_id] = \
                    SceneManager.INVALID_POINT3D

    #---------------------------------------------------------------------------

    # camera_list: set of cameras whose points we'd like to keep
    # min/max triangulation angle: in degrees
    def filter_points3D(self,
            min_track_len=0, max_error=np.inf, min_tri_angle=0,
            max_tri_angle=180, image_set=set()):

        image_set = set(image_set)

        check_triangulation_angles = (min_tri_angle > 0 or max_tri_angle < 180)
        if check_triangulation_angles:
            max_tri_prod = np.cos(np.radians(min_tri_angle))
            min_tri_prod = np.cos(np.radians(max_tri_angle))

        iter_point3D_id_to_point3D_idx = \
            self.point3D_id_to_point3D_idx.iteritems()

        image_ids = []

        for point3D_id, point3D_idx in iter_point3D_id_to_point3D_idx:
            if point3D_idx == SceneManager.INVALID_POINT3D:
                continue

            if image_set or min_track_len > 0:
                image_ids = set(self.point3D_id_to_images[point3D_id][:,0])
            
            # check if error and min track length are sufficient, or if none of
            # the selected cameras see the point
            if (len(image_ids) < min_track_len or
                      self.point3D_errors[point3D_idx] > max_error or
                      image_set and image_set.isdisjoint(image_ids)):
                self.point3D_id_to_point3D_idx[point3D_id] = \
                    SceneManager.INVALID_POINT3D

            # find dot product between all camera viewing rays
            elif check_triangulation_angles:
                xyz = self.points3D[point3D_idx,:]
                tvecs = np.array(
                    [(self.images[image_id].tvec - xyz)
                     for image_id in image_ids])
                tvecs /= np.linalg.norm(tvecs, axis=-1)[:,np.newaxis]

                cos_theta = np.array(
                    [u.dot(v) for u,v in combinations(tvecs, 2)])

                # min_prod = cos(maximum viewing angle), and vice versa
                # if maximum viewing angle is too small or too large,
                # don't add this point
                if (np.min(cos_theta) > max_tri_prod or
                    np.max(cos_theta) < min_tri_prod):
                    self.point3D_id_to_point3D_idx[point3D_id] = \
                        SceneManager.INVALID_POINT3D

        # apply the filters to the image point3D_ids
        for image in self.images.itervalues():
            mask = np.array([
                self.point3D_id_to_point3D_idx.get(point3D_id, 0) \
                    == SceneManager.INVALID_POINT3D
                for point3D_id in image.point3D_ids])
            image.point3D_ids[mask] = SceneManager.INVALID_POINT3D

    #---------------------------------------------------------------------------

    # scene graph: {image_id: [image_id: #shared points]}
    def build_scene_graph(self):
        self.scene_graph = defaultdict(lambda: defaultdict(int))
        point3D_iter = self.point3D_id_to_images.iteritems()

        for i, (point3D_id, images) in enumerate(point3D_iter):
            if not self.point3D_valid(point3D_id):
                continue

            for image_id1, image_id2 in combinations(images[:,0], 2):
                self.scene_graph[image_id1][image_id2] += 1
                self.scene_graph[image_id2][image_id1] += 1

    
if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
