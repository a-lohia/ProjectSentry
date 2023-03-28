from pathlib import Path
from typing import Union, List


class UFPRPlateDataset:

    def __init__(self, dataset_dir: Path):
        # Dictionary[path to image, tuple(licence plate number, plate bounding box (x_min, y_min, x_max, y_max))


    def read_image_txt(self, image_id: int, ) -> Union[List, tuple]:

        with open("track0091[01].txt", "r") as track0091_01_txt:
            txt = track0091_01_txt.readlines()

            lp = txt[6].replace("plate: ", "")  # line 6 in the txt has licence plate number
            plate = txt[7]  # line 7 has bounding box location for licence plate

            plate = ''.join(i for i in plate if i.isdigit() or i == " ")
            temp_list = [int(i) for i in plate.replace(" ", "", 1).split(" ")]
            # plate_date = (x_min, y_min, x_max, y_max)
            plate_data = (temp_list[0], temp_list[1], temp_list[0] + temp_list[2], temp_list[1] + temp_list[3])

        return list(lp)[:-1], plate_data
