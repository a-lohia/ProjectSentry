from pathlib import Path
from typing import Union, List, Dict
import os
import re
import glob

import PIL
import numpy as np
import torchvision
from matplotlib import patches
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt


def read_image_txt(image_id: str, ) -> Union[List, tuple]:
    """
    :param image_id: in the format of: "track####[##].txt"
    :return: Union[List(Licence Plate Characters), Plate_Boundary: Tuple(Xmin, Ymin, Xmax, Ymax)]
    """
    # print(image_id)
    with open(image_id, "r") as txt_file:
        txt = txt_file.readlines()

        lp = txt[6].replace("plate: ", "")  # line 6 in the txt has licence plate number
        plate = txt[7]  # line 7 has bounding box location for licence plate

        plate = ''.join(i for i in plate if i.isdigit() or i == " ")
        # print(plate.replace(" ", "", 1).split(" "))

        # removing all extra spaces at the end
        while plate[-1] == " ":
            plate = plate[:-1]

        # convert all the numbers into list items except the first space
        temp_list = [int(i) for i in plate.replace(" ", "", 1).split(" ")]
        # plate_date = (x_min, y_min, x_max, y_max)
        plate_data = np.array([temp_list[0], temp_list[1], temp_list[0] + temp_list[2], temp_list[1] + temp_list[3]],
                              dtype=np.int32)
    # print(f"{lp[:-1]} {plate_data}")
    return list(lp)[:-1], plate_data


def id_to_filepath(_id: str) -> str:
    """
    returns the absolute filepath (str) of a photo with 6-digit id ("XXXXUU") for use during training
    :param _id: string "XXXXUU"
    :return file: string absolute filepath
    """
    assert (len(_id) == 6)
    track = _id[:4]
    photo_num = _id[4:]
    file = glob.glob(
        f"C:\\Users\\Arya\\workspace\\ProjectSentry\\data\\raw\\UFPR-ALPR dataset/**/*{track}[[]{photo_num}[]].png",
        recursive=True
    )[0]

    return file


def annotate_frame_with_bb(image: PIL.Image, bounding_box, *args):
    """
    :param image: PIL Image
    :param bounding_box: Tuple(x_min, y_min, x_max, y_max)
    :param args: torchvision.transforms."transform" (i.e. resize or grayscale)
    :return: fig, ax
    """

    for T in args:
        image = Image.fromarray(np.asarray(T(image)))

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    image = torch.tensor(np.asarray(image), dtype=torch.float32)

    bb = bounding_box  # model(image[None, :]).detach().mean(axis=0)
    x_min, y_min, x_max, y_max = bb
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=.5,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)

    return fig, ax


def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    # print(bb)
    rows, cols, *_ = x.shape
    Y = np.zeros((rows, cols))
    # bb = bb.astype(np.int)
    Y[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1] = 1
    return Y


def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    x_min = np.min(rows)
    y_min = np.min(cols)
    x_max = np.max(rows)
    y_max = np.max(cols)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)


def resize_image_bb(image: PIL.Image, bb, sz):
    # write_path,
    """Resize an image and its bounding box and write image to new path"""
    im = np.asarray(image)
    # print(im.shape)
    im_resized = cv2.resize(im, (int((16 / 9) * sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int((16 / 9) * sz), sz))
    # new_path = str(write_path/read_path.parts[-1])
    # cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return im_resized, mask_to_bb(Y_resized)


'''class UFPRPlateDataset:
    """
    Class to make accessing data from UFPR Plate Dataset easy
    """

    def __init__(self, dataset_dir):
        # Dictionary[path to image, tuple(licence plate number, plate bounding box (x_min, y_min, x_max, y_max))
        # print(dataset_dir)
        # print(os.path.join(dataset_dir, "testing"))
        self.paths = {
            "testing": os.path.join(dataset_dir, "testing"),
            "training": os.path.join(dataset_dir, "training"),
            "validation": os.path.join(dataset_dir, "validation")
        }

        # Image_Id: Features of the ID
        self.training_set = {}
        self.testing_set = {}
        self.validation_set = {}

        print(f"setting up on paths: {self.paths}")
        self._setup_training_data()
        self._setup_testing_data()
        self._setup_validation_data()

        print("setup complete. the UFPR-ALPR dataset is now ready for use!")

    def _setup_training_data(self):
        """Run once on initialization of the database object"""
        os.chdir("C:/Users/Arya/workspace/ProjectSentry/data/raw/UFPR-ALPR dataset")

        # dictionary which has values of lists of sets of pictures
        image_sets = {i: os.listdir(j) for i, j in zip(self.paths.keys(), self.paths.values())}

        # each image is labeled in the format:
        # track####[##].png (with an accompanying track####[##].txt

        # filters out the "track" from each folder name
        _training_cars = [re.sub("[^0123456789]", "", i) for i in image_sets["training"]]

        # list of id paths in the training set
        _training_ids = []
        for car in _training_cars:
            _training_ids.extend(os.listdir(os.path.join(self.paths['training'], f"track{car}")))

        # list of the features corresponding to each training id ("XXXXUU" 4-digit number plus 2-digit photo #)
        # for each car
        _training_features = [
            read_image_txt(
                os.path.join(self.paths["training"], f"track{training_id[5:9]}", training_id)
            ) for training_id in _training_ids if (".txt" in training_id)
        ]

        # list of training ids for each car ("XXXXUU" 4-digit number plus 2-digit photo #)
        _training_ids = [re.sub("[^0123456789]", "", i) for i in _training_ids]

        # dictionary that maps training id to features for each photo
        self.training_set = {photo_id: feature for photo_id, feature in zip(_training_ids, _training_features)}
        print("finished setting up training data")

    def _setup_testing_data(self):
        """Run once on initialization of the database object"""
        os.chdir("C:/Users/Arya/workspace/ProjectSentry/data/raw/UFPR-ALPR dataset")

        # dictionary which has values of lists of sets of pictures
        image_sets = {i: os.listdir(j) for i, j in zip(self.paths.keys(), self.paths.values())}

        # each image is labeled in the format:
        # track####[##].png (with an accompanying track####[##].txt

        # filters out the "track" from each folder name
        _testing_cars = [re.sub("[^0123456789]", "", i) for i in image_sets['testing']]

        # list of id paths in the training set
        _testing_ids = []
        for car in _testing_cars:
            _testing_ids.extend(os.listdir(os.path.join(self.paths['testing'], f"track{car}")))

        # list of the features corresponding to each testing id ("XXXXUU" 4-digit number plus 2-digit photo #)
        # for each car
        _testing_features = [
            read_image_txt(
                os.path.join(self.paths["testing"], f"track{testing_id[5:9]}", testing_id)
            ) for testing_id in _testing_ids if (".txt" in testing_id)
        ]

        _testing_ids = [re.sub("[^0123456789]", "", i) for i in _testing_ids]

        # dictionary that maps testing id to features for each photo
        self.testing_set = {photo_id: feature for photo_id, feature in zip(_testing_ids, _testing_features)}
        print("finished setting up testing data")

    def _setup_validation_data(self):
        """Run once on initialization of the database object"""
        os.chdir("C:/Users/Arya/workspace/ProjectSentry/data/raw/UFPR-ALPR dataset")

        # dictionary which has values of lists of sets of pictures
        image_sets = {i: os.listdir(j) for i, j in zip(self.paths.keys(), self.paths.values())}

        # each image is labeled in the format:
        # track####[##].png (with an accompanying track####[##].txt

        # filters out the "track" from each folder name
        _validation_cars = [re.sub("[^0123456789]", "", i) for i in image_sets['validation']]

        # list of id paths in the training set
        _validation_ids = []
        for car in _validation_cars:
            _validation_ids.extend(os.listdir(os.path.join(self.paths['validation'], f"track{car}")))

        # list of the features corresponding to each validation id ("XXXXUU" 4-digit number plus 2-digit photo #)
        # for each car
        _validation_features = [
            read_image_txt(
                os.path.join(self.paths["validation"], f"track{validation_id[5:9]}", validation_id)
            ) for validation_id in _validation_ids if (".txt" in validation_id)
        ]

        _validation_ids = [re.sub("[^0123456789]", "", i) for i in _validation_ids]

        # dictionary that maps validation id to features for each photo
        self.testing_set = {photo_id: feature for photo_id, feature in zip(_validation_ids, _validation_features)}
        print("finished setting up validation data")
'''


# Good one. Use with Pytorch Dataloader. Don't use the previous one

class UFPRDataset(Dataset):
    def __init__(self, dataset_dir, grayscale=None, resize=None):
        self.resize = resize
        self.grayscale = grayscale
        self.dataset_dir = dataset_dir

        # Dict[int(Image_Id) -> Tuple(Licence Plate #, Licence Plate Coords)]
        self.ids = []
        self.labels = {}

        self._setup()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = id_to_filepath(self.ids[idx])
        image = np.asarray(Image.open(img_path))
        label = self.labels[str(self.ids[idx])]

        if self.resize:
            # computes new label (new bounding box coords) (from old image size) then resizes image
            label = label[0], mask_to_bb(self.resize(Image.fromarray(create_mask(label[1], image))))
            image = np.asarray(self.resize(Image.fromarray(image)))

        if self.grayscale:
            image = np.asarray(self.grayscale(Image.fromarray(image)))

        image = torch.tensor(image, dtype=torch.float32)

        return image, label

    def _setup(self):
        os.chdir(self.dataset_dir)

        # each image is labeled in the format:
        # track####[##].png (with an accompanying track####[##].txt

        # filters out the "track" from each folder name
        _training_cars = [re.sub("[^0123456789]", "", i) for i in (os.listdir(self.dataset_dir))]
        # print(_training_cars)

        # list of id paths in the training set
        _training_ids_and_txt = []

        for car in _training_cars:
            if car.isdigit():
                _training_ids_and_txt.extend(os.listdir(os.path.join(self.dataset_dir, f"track{car}")))
            # print(f"{_training_ids} + \n")

        # list of the features corresponding to each training id ("XXXXUU" 4-digit number plus 2-digit photo #)
        # for each car
        _training_features = [
            read_image_txt(
                os.path.join(self.dataset_dir, f"track{training_id[5:9]}", training_id)
            ) for training_id in _training_ids_and_txt if (".txt" in training_id)
        ]

        _training_ids = [training_id for training_id in _training_ids_and_txt if (".png" in training_id)]

        # list of training ids for each car ("XXXXUU" 4-digit number plus 2-digit photo #)
        _training_ids = [re.sub("[^0123456789]", "", i) for i in _training_ids]

        self.ids = _training_ids
        # dictionary that maps training id to features for each photo
        self.labels = {photo_id: feature for photo_id, feature in zip(_training_ids, _training_features)}
        print("finished setting up data")
