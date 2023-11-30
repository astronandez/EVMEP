import os
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn import preprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_path', default=r'\CompCars\data',
                    help='path to dataset')

parser.add_argument('--annotation_path', default='YoloFormatData/',
                    help='path to save annotation')

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]


def main():
    car_types = pd.read_csv('YoloFormatData/test.csv')
    args = parser.parse_args()
    img_path = os.path.join(r'Z:', args.data_path, 'image/')
    label_path = os.path.join(r'Z:', args.data_path, 'label/')
    type_path = os.path.join(r'Z:', args.data_path, 'misc/')

    filelist = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            filelist.append(os.path.join(root, file))

    labellist = []
    for root, dirs, files in os.walk(label_path):
        for file in files:
            labellist.append(os.path.join(root, file))

    filelist = sorted(filelist)
    labellist = sorted(labellist)

    # filelist = filelist[:1000]
    # labellist = labellist[:1000]

    img_w = []
    img_h = []

    for i in tqdm(range(len(filelist))):
        current_image = cv.imread(f"{filelist[i]}")
        w, h = current_image.shape[:2]
        img_w.append(w)
        img_h.append(h)
        # if i == 1000:
        #     break

    filelist = [x.replace(img_path, '') for x in filelist]
    full_data_single = pd.DataFrame(filelist, columns=['image_name'])
    full_data = full_data_single['image_name'].str.split('\\', expand=True)
    full_data.columns = ['make_id', 'model_id', 'year', 'image_name']

    full_data['make_id'] = pd.to_numeric(full_data['make_id'], errors='coerce', downcast='integer')
    full_data['model_id'] = pd.to_numeric(full_data['model_id'], errors='coerce', downcast='integer')
    full_data['year'] = pd.to_numeric(full_data['year'], errors='coerce', downcast='integer')

    car_types = car_types.drop(['maximum_speed', 'displacement', 'door_number', 'seat_number'], axis=1)
    car_types = car_types.pivot_table(index=car_types.index, columns='type', values='model_id', aggfunc='first')

    num_row = full_data.shape[0]
    lbs = pd.DataFrame({'car_type': [0] * num_row})
    model_ray = np.array(full_data['model_id'].tolist())

    for index, value in tqdm(enumerate(model_ray)):
        for j in range(12):
            if value in car_types[j+1].values:
                lbs.at[index, 'car_type'] = j+1

        # if index == 1000:
        #     break

    full_data['car_type'] = lbs['car_type']
    print(full_data.head())

    x1s = []
    y1s = []
    x2s = []
    y2s = []

    print('Read annotations')
    for i in tqdm(range(len(labellist))):
        result = pd.read_csv(labellist[i], header=None).loc[2].values[0].split(' ')
        result = [int(x) for x in result]
        x1s.append(result[0])
        y1s.append(result[1])
        x2s.append(result[2])
        y2s.append(result[3])
        # if i == 1000:
        #     break

    x1_yolo = []
    y1_yolo = []
    x2_yolo = []
    y2_yolo = []

    for i in tqdm(range(len(x2s))):
        yolo_coords = pascal_voc_to_yolo(x1s[i], y1s[i], x2s[i], y2s[i], img_w[i], img_h[i])
        x1_yolo.append(yolo_coords[0])
        y1_yolo.append(yolo_coords[1])
        x2_yolo.append(yolo_coords[2])
        y2_yolo.append(yolo_coords[3])
        # if i == 1000:
        #     break

    full_data['x_1'] = x1_yolo
    full_data['y_1'] = y1_yolo
    full_data['x_2'] = x2_yolo
    full_data['y_2'] = y2_yolo
    full_data['image_w'] = img_w
    full_data['image_h'] = img_h

    yolo_full_data = full_data[['image_name', 'car_type', 'x_1', 'y_1', 'x_2', 'y_2']]
    # yolo_full_data = full_data[['car_type', 'x_1', 'y_1', 'x_2', 'y_2']]

    if not os.path.isdir(args.annotation_path):
        os.mkdir(args.annotation_path)

    yolo_full_data.to_csv(args.annotation_path + 'yolo_formated_data.txt', index=False)

if __name__ == '__main__':
    main()