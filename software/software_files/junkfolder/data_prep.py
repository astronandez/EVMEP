import os
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn import preprocessing
from tqdm import tqdm
from torch import tensor
from torchvision.ops import box_convert
import pybboxes as pbx

#For debugging set debug_mode to True and the num_to_test equal to the number of entries desired
debug_mode = False
num_to_test = 10

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_path', default=r'\CompCars\data',
                    help='path to dataset')

parser.add_argument('--annotation_path', default='YoloFormatData/',
                    help='path to save annotation')

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    voc_box = (x1, y1, x2, y2)
    yolobox = pbx.convert_bbox(voc_box, from_type="voc", to_type="yolo", image_size=[image_w, image_h])
    # return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]
    return yolobox

def voc_to_yolo(xyxy, image_w, image_h):
    yolobox = pbx.convert_bbox(xyxy, from_type="voc", to_type="yolo", image_size=[image_w, image_h])
    return yolobox

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

    if(debug_mode):
        filelist = filelist[:num_to_test]
        labellist = labellist[:num_to_test]

    x1_yolo = []
    y1_yolo = []
    x2_yolo = []
    y2_yolo = []
    junk_image_pile = []
    junk_label_pile = []

    for i in tqdm(range(len(filelist))):
        current_image = cv.imread(f"{filelist[i]}")
        h, w, _ = current_image.shape
        result = pd.read_csv(labellist[i], header=None).loc[2].values[0].split(' ')
        x1, y1, x2, y2 = map(float, result)
        if (x1 > 0 and x2 > x1 and x2 <= w and
            y1 > 0 and y2 > y1 and y2 <= h):
            diffw = x2 - x1
            diffh = y2 - y1
            cx = ((x1 + (diffw/2)) / w)
            cy = ((y1 + (diffh/2)) / h)
            nw = ((diffw) / w)
            nh = ((diffh) / h)
            x1_yolo.append(cx)
            y1_yolo.append(cy)
            x2_yolo.append(nw)
            y2_yolo.append(nh)
        else:
            junk_image_pile.append(filelist[i])
            junk_label_pile.append(labellist[i])

    for i in tqdm(range(len(junk_image_pile))):
        filelist.remove(junk_image_pile[i])
        labellist.remove(junk_label_pile[i])

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

    full_data['car_type'] = lbs['car_type']

    full_data['x_1'] = x1_yolo
    full_data['y_1'] = y1_yolo
    full_data['x_2'] = x2_yolo
    full_data['y_2'] = y2_yolo

    yolo_full_data = full_data[['image_name', 'car_type', 'x_1', 'y_1', 'x_2', 'y_2']]
    yolo_img_data = full_data['image_name']

    if not os.path.isdir(args.annotation_path):
        os.mkdir(args.annotation_path)

    yolo_full_data.to_csv(args.annotation_path + 'yolo_formated_data.txt', index=False)
    yolo_img_data.to_csv(args.annotation_path + 'yolo_image_data.txt', index=False)

if __name__ == '__main__':
    main()