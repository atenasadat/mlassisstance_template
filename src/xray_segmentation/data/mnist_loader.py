from typing import TYPE_CHECKING, Tuple, Union
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Type, Union, TYPE_CHECKING, Set, Any

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mlassistant.core.data import ContentLoader

if TYPE_CHECKING:
    from ..config import MnistConfig
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
class MnistLoader(ContentLoader):
    """ The MnistLoader class """

    def __init__(self, conf: 'MnistConfig', prefix_name: str, data_specification: str):
        super().__init__(conf, prefix_name, data_specification)
        self._x, self._y = self._load_data(data_specification)

        # self.class_samples_indices: Dict[Any, np.ndarray] = dict()
        # for i in range(5):
        #     if self._y[i] not in self.class_samples_indices:
        #           print("hi")
        #     self.class_samples_indices[self._y[i]] = []
        # self.class_samples_indices[self._y[i]].append(i)

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

                # filename, fileext = os.path.splitext(base_file)
                # cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, base_file), image)
                # cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, "%s_mask%s" % (filename, fileext)), mask)
                # cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, "%s_dilate%s" % (filename, fileext)), mask_dilate)

            else:
                # print("Monto Validation")
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_DILATE_DIR, base_file), mask_dilate)
                # print("validation")
                # filename, fileext = os.path.splitext(base_file)
                # cv2.imwrite(os.path.join(SEGMENTATION_VALID_DIR, base_file), image)
                # cv2.imwrite(os.path.join(SEGMENTATION_VALID_DIR, "%s_mask%s" % (filename, fileext)), mask)
                # cv2.imwrite(os.path.join(SEGMENTATION_VALID_DIR, "%s_dilate%s" % (filename, fileext)), mask_dilate)

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
                print("Shenzen train")
                cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base_file), mask_dilate)
            elif (mask_file in shenzhen_test):
                # print("Shenzen test")
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_TEST_DILATE_DIR, base_file), mask_dilate)

                # filename, fileext = os.path.splitext(base_file)

                # cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, base_file), \
                #             image)
                # cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, \
                #                         "%s_mask%s" % (filename, fileext)), mask)
                # cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, \
                #                         "%s_dilate%s" % (filename, fileext)), mask_dilate)


            else:

                # print("Shenzen Validation")
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_IMAGE_DIR, base_file), image)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_MASK_DIR, base_file), mask)
                cv2.imwrite(os.path.join(SEGMENTATION_VALID_DILATE_DIR, base_file), mask_dilate)
                # print("valid shenzen")
                # filename , fileext = os.path.splitext(base_file)
                # cv2.imwrite(os.path.join(SEGMENTATION_VALID_DIR, base_file), \
                #             image)
                # cv2.imwrite(os.path.join(SEGMENTATION_VALID_DIR, \
                #                         "%s_mask%s" % (filename, fileext)), mask)
                # cv2.imwrite(os.path.join(SEGMENTATION_VALID_DIR, \
                #                         "%s_dilate%s" % (filename, fileext)), mask_dilate)

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

        print("x_train", type(X_train), len(X_train))
        print("x_train[0]", (cv2.imread(X_train[0])).shape)  # 512 512 3

        # x_train , x_test , x_val= np.empty(len(X_train)) , np.empty(len(X_test)) , np.empty(len(X_val))
        # y_train , y_test , y_val= np.empty(len(Y_train)) , np.empty(len(Y_test)) , np.empty(len(Y_val))
        x_train, x_test, x_val = [], [], []
        y_train, y_test, y_val = [], [], []
        for i in range(5):
            x_train.append(cv2.imread(X_train[i]).reshape(3, 512, 512))
            y_train.append(cv2.imread(Y_train[i]).reshape(3, 512, 512))
        for i in range(5):
            x_val.append(cv2.imread(X_val[i]).reshape(3, 512, 512))
            y_val.append(cv2.imread(Y_val[i]).reshape(3, 512, 512))
        for i in range(5):
            x_test.append(cv2.imread(X_test[i]).reshape(3, 512, 512))
            y_test.append(cv2.imread(Y_test[i]).reshape(3, 512, 512))

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(x_train.shape)

        data = {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test),
        }

        return data[data_specification]

    def get_samples_names(self):
        ''' sample names must be unique, they can be either scan_names or scan_dirs.
        Decided to put scan_names. No difference'''
        # print("sample names" , [str(i) for i in range(len(self._x))])
        return [str(i) for i in range(len(self._x))]

    def get_samples_labels(self):
        # print("get sample labels:-------" , self._y[0] )
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
        }

    def _get_x(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray]) \
            -> np.ndarray:
        return self._x[samples_inds]

    def _get_y(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray]) \
            -> np.ndarray:
        return self._y[samples_inds]
