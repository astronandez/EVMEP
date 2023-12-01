import os
import shutil
import time
from tqdm import tqdm

def reformat_directory(source_dir, destination_dir, folder_name, file_based=True):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    new_subfolder = os.path.join(destination_dir, folder_name)

    if not os.path.exists(new_subfolder):
        os.makedirs(new_subfolder)

    # count = 0
    # for root, dirs, files in os.walk(source_dir):
    #     count += len(files)
    # progress_bar = tqdm(total=count, desc="Organizing")
    if file_based:
        # progress_bar = tqdm(total=count, desc="Organizing Images")
        for root, dirs, files in os.walk(source_dir):
            relative_path = os.path.relpath(root, source_dir)

            for file in files:
                source_file = os.path.join(root, file)
                destination_file = os.path.join(new_subfolder, f"{file}")

                shutil.copy2(source_file, destination_file)

        #     progress_bar.update(1)
        # progress_bar.close()

    else:
        with open(source_dir, 'r') as file:
            # Skip the first line
            next(file)

            # Process each subsequent line
            for line in file:
                column = [line.strip().split(',')]
                column[0][2:6] = [float(x) for x in column[0][1:6]]
                column[0][0] = column[0][0].replace(".jpg", ".txt")
                new_file_name = f'{column[0][0]}'

                string_data = f'{column[0][1]}' + ' ' + f'{column[0][2]}' + ' ' + f'{column[0][3]}' + ' ' + f'{column[0][4]}' + ' ' + f'{column[0][5]}'

                file_dest = os.path.join(new_subfolder, new_file_name)
                with open(file_dest, 'w') as out_file:
                    out_file.write(string_data)



if __name__ == "__main__":
    source_directory = r'Z:\CompCars\data\image'
    destination_directory = r'Z:\CompCarsYOLO\data'
    folder_name = 'images'

    source_directory2 = 'YoloFormatData/yolo_formated_data.txt'
    destination_directory2 = r'Z:\CompCarsYOLO\data'
    folder_name2 = 'labels'

    reformat_directory(source_directory, destination_directory, folder_name)
    reformat_directory(source_directory2, destination_directory2, folder_name2, file_based=False)