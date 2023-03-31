from pathlib import Path
from typing import Union, List, Dict
import os


class UFPRPlateDataset:
    """
    Class to make accessing data from UFPRPlateDataset easy
    """

    def __init__(self, dataset_dir: Path):
        # Dictionary[path to image, tuple(licence plate number, plate bounding box (x_min, y_min, x_max, y_max))
        self.paths = {
            "testing": os.path.join(dataset_dir, "/testing"),
            "training": os.path.join(dataset_dir, "/training"),
            "validation": os.path.join(dataset_dir, "/validation")
        }


    def _setup(self):
        """Run once on initialization of the database object"""

        # dictionary which has values of lists of sets of pictures
        image_sets = {j: os.listdir(i) for i, j in zip(self.paths.values(), self.paths.keys())}



    def read_image_txt(self, image_id: str, ) -> Union[List, tuple]:
        """
        :param image_id: in the format of: "track####[##].txt"
        :return: Union[List(Licence Plate Characters), Plate_Boundary: Tuple(Xmin, Ymin, Xmax, Ymax)]
        """

        with open(image_id, "r") as txt_file:
            txt = txt_file.readlines()

            lp = txt[6].replace("plate: ", "")  # line 6 in the txt has licence plate number
            plate = txt[7]  # line 7 has bounding box location for licence plate

            plate = ''.join(i for i in plate if i.isdigit() or i == " ")
            temp_list = [int(i) for i in plate.replace(" ", "", 1).split(" ")]
            # plate_date = (x_min, y_min, x_max, y_max)
            plate_data = (temp_list[0], temp_list[1], temp_list[0] + temp_list[2], temp_list[1] + temp_list[3])

        return list(lp)[:-1], plate_data
