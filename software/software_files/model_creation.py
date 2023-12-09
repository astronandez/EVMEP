import cv2
import sys
import numpy as np
import torch
import onnxruntime as rt
import onnx
from torch.utils.data import DataLoader
from super_gradients.training import models
from super_gradients.training.utils.media.image import load_image
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.models import model_factory
from super_gradients.conversion import DetectionOutputFormatMode
from super_gradients.conversion import ExportQuantizationMode
from visualization_model_predictions import show_predictions_from_flat_format
def load_checkpoint(model, ckpt_file):
  state_dict = torch.load(ckpt_file, map_location="cuda")
  ckpt_key = "ema_net" if "ema_net" in state_dict else "net"
  adaptive_load_state_dict(model, state_dict[ckpt_key], strict="no_key_matching")

def create_calibration_data():
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VAL_IMAGES_DIR,
            'labels_dir': config.VAL_LABELS_DIR,
            'classes': config.CLASSES,
            'input_dim': [320, 320]
        },
        dataloader_params=config.DATALOADER_PARAMS
    )
    return val_data

def get_our_model():
    best_model = models.get(config.MODEL_NAME,
                            num_classes=config.NUM_CLASSES,
                            checkpoint_path='Z:/CompCarsYOLO/model/content/checkpoints/yolo_car_type_identifier_qat/RUN_20231207_135514_373944/ckpt_best.pth')
    return best_model

def ex_model(val_data):
    export_result = best_model.export(
        "yolo_nas_s_int8_with_calibration_v2.onnx",
        confidence_threshold = 0.4,
        nms_threshold = 0.5,
        num_pre_nms_predictions = 100,
        max_predictions_per_image = 1,
        output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
        quantization_mode=ExportQuantizationMode.INT8,
        calibration_loader=val_data
    )
    return export_result


class config:
    #trainer params
    CHECKPOINT_DIR = 'Z:/CompCarsYOLO/model/content/checkpoints' #specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'yolo_car_type_identifier_qat' #specify the experiment name

    #dataset params
    DATA_DIR = 'Z:/CompCarsYOLO/model/content/vehicle-body-style-dataset-2-yolov5/' #parent directory to where data lives
    DATA_NAME = 'vehicle-body-style-dataset-2-yolov5'

    TRAIN_IMAGES_DIR = 'train/images' #child dir of DATA_DIR where train images are
    TRAIN_LABELS_DIR = 'train/labels' #child dir of DATA_DIR where train labels are

    VAL_IMAGES_DIR = 'valid/images' #child dir of DATA_DIR where validation images are
    VAL_LABELS_DIR = 'valid/labels' #child dir of DATA_DIR where validation labels are

    # if you have a test set
    TEST_IMAGES_DIR = 'test/images' #child dir of DATA_DIR where test images are
    TEST_LABELS_DIR = 'test/labels' #child dir of DATA_DIR where test labels are

    CLASSES = ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'estate',
               'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible'] #what class names do you have

    NUM_CLASSES = len(CLASSES)

    #dataloader params - you can add whatever PyTorch dataloader params you have
    #could be different across train, val, and test
    DATALOADER_PARAMS = {
        'batch_size':16,
        'num_workers':2
    }

    # model params
    MODEL_NAME = 'yolo_nas_s' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco' #only one option here: coco

val_data = create_calibration_data()
best_model = get_our_model()

if __name__ == '__main__':
    export_result = ex_model(val_data)
    print(export_result.output, file=sys.stderr)

# image = load_image('Z:/CompCarsYOLO/model/content/single_vehicle_test.jpg')
# image = cv2.resize(image, (export_result.input_image_shape[1], export_result.input_image_shape[0]))
# image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
#
# session = rt.InferenceSession(export_result.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# inputs = [o.name for o in session.get_inputs()]
# outputs = [o.name for o in session.get_outputs()]
# result = session.run(outputs, {inputs[0]: image_bchw})
#
# show_predictions_from_flat_format(image, result)
