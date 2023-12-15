import os
import shutil
import time
from tqdm import tqdm

def is_name_in_file(file_path, target_name):
    with open(file_path, 'r') as file:
        for line in file:
            if target_name in line:
                return True
    return False

def reformat_directory(source_dir, destination_dir, folder_name, img_source, img_text):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    new_subfolder = os.path.join(destination_dir, folder_name)
    new2_subfolder = os.path.join(destination_dir, 'images')

    if not os.path.exists(new_subfolder):
        os.makedirs(new_subfolder)

    if not os.path.exists(new2_subfolder):
        os.makedirs(new2_subfolder)

    for root, dirs, files in os.walk(img_source):
        for file in files:
            source_file = os.path.join(root, file)
            if is_name_in_file(img_text, f"{file}"):
                destination_file = os.path.join(new2_subfolder, f"{file}")
                shutil.copy2(source_file, destination_file)

    with open(source_dir, 'r') as file:
        # Skip the first line
        next(file)

        # Process each subsequent line
        for line in file:
            column = [line.strip().split(',')]
            column[0][2:6] = [float(x) for x in column[0][2:6]]
            column[0][0] = column[0][0].replace(".jpg", ".txt")
            new_file_name = f'{column[0][0]}'

            string_data = f'{column[0][1]}' + ' ' + f'{column[0][2]}' + ' ' + f'{column[0][3]}' + ' ' + f'{column[0][4]}' + ' ' + f'{column[0][5]}'
            file_dest = os.path.join(new_subfolder, new_file_name)
            with open(file_dest, 'w') as out_file:
                out_file.write(string_data)



if __name__ == "__main__":
    img_source = r'Z:\CompCars\data\image'
    img_text = 'YoloFormatData/yolo_image_data.txt'
    source_directory2 = 'YoloFormatData/yolo_formated_data.txt'
    destination_directory2 = r'Z:\CompCarsYOLO\data'
    folder_name2 = 'labels'

    reformat_directory(source_directory2, destination_directory2, folder_name2, img_source, img_text)