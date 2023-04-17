from pathlib import Path
from typing import Union, List, Dict
import os
import re
import glob
import numpy as np


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
        plate_data = np.array([temp_list[0], temp_list[1], temp_list[0] + temp_list[2], temp_list[1] + temp_list[3]], dtype=np.int32)

    return list(lp)[:-1], plate_data


def id_to_filepath(_id: str) -> str:
    """
    returns the absolute filepath (str) of a photo with 6-digit id ("XXXXUU") for use during training
    :param _id: string "XXXXUU"
    :return file: string absolute filepath
    """
    assert(len(_id) == 6)
    track = _id[:4]
    photo_num = _id[4:]
    file = glob.glob(
        f"C:\\Users\\Arya\\workspace\\ProjectSentry\\data\\raw\\UFPR-ALPR dataset/**/*{track}[[]{photo_num}[]].png",
        recursive=True
    )[0]

    return file

# TODO: Implement this into a pytorch Database https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class UFPRPlateDataset:
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



