from pathlib import Path
from typing import Union, List, Dict
import os
import re


def read_image_txt(image_id: str, ) -> Union[List, tuple]:
    """
    :param image_id: in the format of: "track####[##].txt"
    :return: Union[List(Licence Plate Characters), Plate_Boundary: Tuple(Xmin, Ymin, Xmax, Ymax)]
    """
    print(image_id)
    with open(image_id, "r") as txt_file:
        txt = txt_file.readlines()

        lp = txt[6].replace("plate: ", "")  # line 6 in the txt has licence plate number
        plate = txt[7]  # line 7 has bounding box location for licence plate

        plate = ''.join(i for i in plate if i.isdigit() or i == " ")
        temp_list = [int(i) for i in plate.replace(" ", "", 1).split(" ")]
        # plate_date = (x_min, y_min, x_max, y_max)
        plate_data = (temp_list[0], temp_list[1], temp_list[0] + temp_list[2], temp_list[1] + temp_list[3])

    return list(lp)[:-1], plate_data


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

        self._setup_training_data()

    def _setup_training_data(self):
        """Run once on initialization of the database object"""
        print(self.paths)
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

        _training_features = [
            read_image_txt(
                os.path.join(self.paths["training"], f"track{training_id[5:9]}", training_id)
            ) for training_id in _training_ids if (".txt" in training_id)
        ]

        _training_ids = [re.sub("[^0123456789]", "", i) for i in _training_ids]

        self.training_set = {photo_id: feature for photo_id, feature in zip(_training_ids, _training_features)}


