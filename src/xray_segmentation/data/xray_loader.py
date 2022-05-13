from typing import TYPE_CHECKING, Tuple, Union
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Type, Union, TYPE_CHECKING, Set, Any

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mlassistant.core.data import ContentLoader

if TYPE_CHECKING:
    from ..config import XRayConfig
import cv2

import os

INPUT_DIR = '/content/drive/MyDrive/xray_segmentation/x_ray_data'
# os.path.join("..", "x_ray_data")

SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentation")
SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
SEGMENTATION_VALID_DIR = os.path.join(SEGMENTATION_DIR, "valid")

SEGMENTATION_AUG_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "augmentation")
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")
SEGMENTATION_DILATE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "dilate")

# SEGMENTATION_VALID_AUG_DIR = os.path.join(SEGMENTATION_VALID_DIR, "augmentation")
SEGMENTATION_VALID_IMAGE_DIR = os.path.join(SEGMENTATION_VALID_DIR, "image")
SEGMENTATION_VALID_MASK_DIR = os.path.join(SEGMENTATION_VALID_DIR, "mask")
SEGMENTATION_VALID_DILATE_DIR = os.path.join(SEGMENTATION_VALID_DIR, "dilate")

# SEGMENTATION_TEST_AUG_DIR = os.path.join(SEGMENTATION_TEST_DIR, "augmentation")
SEGMENTATION_TEST_IMAGE_DIR = os.path.join(SEGMENTATION_TEST_DIR, "image")
SEGMENTATION_TEST_MASK_DIR = os.path.join(SEGMENTATION_TEST_DIR, "mask")
SEGMENTATION_TEST_DILATE_DIR = os.path.join(SEGMENTATION_TEST_DIR, "dilate")

SEGMENTATION_SOURCE_DIR = os.path.join(INPUT_DIR, "pulmonary-chest-xray-abnormalities")

SHENZHEN_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "images")
SHENZHEN_IMAGE_DIR = os.path.join(SHENZHEN_TRAIN_DIR, "images")
SHENZHEN_MASK_DIR = os.path.join(INPUT_DIR, "shcxr-lung-mask", "mask", "mask")

MONTGOMERY_TRAIN_DIR = os.path.join(SEGMENTATION_SOURCE_DIR, "Montgomery", "MontgomerySet")
MONTGOMERY_IMAGE_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "CXR_png")
MONTGOMERY_LEFT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "ManualMask", "leftMask")
MONTGOMERY_RIGHT_MASK_DIR = os.path.join(MONTGOMERY_TRAIN_DIR, "ManualMask", "rightMask")


# print("DataLoader .....")
class XRayLoader(ContentLoader):

    def __init__(self, conf: 'XRayConfig', prefix_name: str, data_specification: str):
        super().__init__(conf, prefix_name, data_specification)
        self._x, self._y, self._mask = self._load_data(data_specification)

    def load_monto_data(self):

        montgomery_left_mask_dir = glob(os.path.join(MONTGOMERY_LEFT_MASK_DIR, '*.png'))
        montgomery_test = montgomery_left_mask_dir[0:50]
        montgomery_train = montgomery_left_mask_dir[50:98]
        montgomery_valid = montgomery_left_mask_dir[98:]

        for left_image_file in tqdm(montgomery_left_mask_dir):
            base_file = os.path.basename(left_image_file)
            image_file = os.path.join(MONTGOMERY_IMAGE_DIR, base_file)
            right_image_file = os.path.join(MONTGOMERY_RIGHT_MASK_DIR, base_file)

            image = cv2.imread(image_file)
            left_mask = cv2.imread(left_image_file, cv2.IMREAD_GRAYSCALE)
            right_mask = cv2.imread(right_image_file, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (512, 512))
            left_mask = cv2.resize(left_mask, (512, 512))
            right_mask = cv2.resize(right_mask, (512, 512))
            DILATE_KERNEL = np.ones((15, 15), np.uint8)

            mask = np.maximum(left_mask, right_mask)
            mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)

            if (left_image_file in montgomery_train):
                cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), mask_dilate)

            elif (left_image_file in montgomery_test):
                # print("Monto test")
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_DILATE_DIR, base_file), mask_dilate)

            else:
                # print("Monto Validation")
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_DILATE_DIR, base_file), mask_dilate)

    def load_shenzen_data(self):
        # print("Loading shenzen data")
        DILATE_KERNEL = np.ones((15, 15), np.uint8)

        shenzhen_mask_dir = glob(os.path.join(SHENZHEN_MASK_DIR, '*.png'))
        shenzhen_test = shenzhen_mask_dir[0:50]
        shenzhen_train = shenzhen_mask_dir[50:450]
        shenzhen_valid = shenzhen_mask_dir[450:]

        for mask_file in tqdm(shenzhen_mask_dir):
            base_file = os.path.basename(mask_file).replace("_mask", "")
            image_file = os.path.join(SHENZHEN_IMAGE_DIR, base_file)

            image = cv2.imread(image_file)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (512, 512))
            mask = cv2.resize(mask, (512, 512))
            mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)

            if (mask_file in shenzhen_train):
                cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), mask_dilate)
            elif (mask_file in shenzhen_test):
                # print("Shenzen test")
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_DILATE_DIR, base_file), mask_dilate)


            else:

                # print("Shenzen Validation")
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_DILATE_DIR, base_file), mask_dilate)

    def _load_data(self, data_specification: str) -> Tuple[np.ndarray, np.ndarray]:

        # self.load_monto_data()
        print("Monto data loaded")
        # self.load_shenzen_data()
        print("Shenzen data loaded")

        # print(len(glob(os.path.join(SHENZHEN_MASK_DIR, '*.png'))))
        # print(len(glob(os.path.join(MONTGOMERY_LEFT_MASK_DIR, '*.png'))))
        X_train = glob(os.path.join(SEGMENTATION_IMAGE_DIR, "*.png"))
        X_test = glob(os.path.join(SEGMENTATION_TEST_IMAGE_DIR, "*.png"))
        X_val = glob(os.path.join(SEGMENTATION_VALID_IMAGE_DIR, "*.png"))

        Y_train = glob(os.path.join(SEGMENTATION_MASK_DIR, "*.png"))
        Y_test = glob(os.path.join(SEGMENTATION_TEST_MASK_DIR, "*.png"))
        Y_val = glob(os.path.join(SEGMENTATION_VALID_MASK_DIR, "*.png"))
        dilate_files = glob(os.path.join(SEGMENTATION_DILATE_DIR, "*.png"))

        # print("x_train" , type(X_train) , len(X_train) )
        # print("x_train[0]",(cv2.imread(X_train[0])).shape) # 512 512 3

        x_train, x_test, x_val = [], [], []
        mask_train, mask_test, mask_val = [], [], []

        print("train data length {} validation {} test {}".format(len(X_train), len(X_val), len(X_test)))
        for i in range(len(X_train)):
            x_train.append(cv2.imread(X_train[i]).reshape(3, 512, 512))
            mask_train.append(cv2.imread(Y_train[i]).reshape(3, 512, 512))
        for i in range(len(X_val)):
            x_val.append(cv2.imread(X_val[i]).reshape(3, 512, 512))
            mask_val.append(cv2.imread(Y_val[i]).reshape(3, 512, 512))
        for i in range(len(X_test)):
            x_test.append(cv2.imread(X_test[i]).reshape(3, 512, 512))
            mask_test.append(cv2.imread(Y_test[i]).reshape(3, 512, 512))
        x_train = np.array(x_train)
        # y_train = np.array(y_train)

        y_train = np.ones(len(x_train))
        y_test = np.ones(len(x_test))
        y_val = np.ones(len(x_val))

        # print(x_train.shape)

        data = {
            'train': (x_train, y_train, mask_train),
            'val': (x_val, y_val, mask_val),
            'test': (x_test, y_test, mask_test),
        }

        return data[data_specification]

    def get_samples_names(self):
        ''' sample names must be unique, they can be either scan_names or scan_dirs.
        Decided to put scan_names. No difference'''
        # print("sample names" , [str(i) for i in range(len(self._x))])
        return [str(i) for i in range(len(self._x))]

    def get_samples_labels(self):
        # print("get sample labels:-------" , self._y[0] )
        # print(self._y)
        # print('hi',self._y)
        return self._y

    def reorder_samples(self, indices, new_names):
        self._x = self._x[indices]
        self._y = self._y[indices]

    def get_views_indices(self):
        return self.get_samples_names(), \
               np.arange(len(self._x)).reshape((len(self._x), 1))

    def get_samples_batch_effect_groups(self):
        pass

    def get_placeholder_name_to_fill_function_dict(self):
        """ Returns a dictionary of the placeholders' names (the ones this content loader supports)
        to the functions used for filling them. The functions must receive as input data_loader,
        which is an object of class data_loader that contains information about the current batch
        (e.g. the indices of the samples, or if the sample has many elements the indices of the chosen
        elements) and return an array per placeholder name according to the receives batch information.
        IMPORTANT: Better to use a fixed prefix in the names of the placeholders to become clear which content loader
        they belong to! Some sort of having a mark :))!"""
        return {
            'x': self._get_x,
            'y': self._get_y,
            'mask': self._get_mask
        }

    def _get_x(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray]) \
            -> np.ndarray:

        return np.array(self._x)[samples_inds]

    def _get_y(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray]) \
            -> np.ndarray:
        return self._y[samples_inds]

    def _get_mask(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray]) \
            -> np.ndarray:
        # print("samples_inds" , type(samples_inds[0]))
        # print("sample_indexes" ,  self._mask[np.array([0, 2])])
        # print("111" , self._mask[0])
        return np.array(self._mask)[samples_inds.astype(int)]









