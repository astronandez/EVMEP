from typing import Tuple, List, Optional
import cv2
import matplotlib.pyplot as plt
from simple_weight_estimation import *

def generate_color_mapping(num_classes: int) -> List[Tuple[int, ...]]:
    """Generate a unique BGR color for each class

    :param num_classes: The number of classes in the dataset.
    :return:            List of RGB colors for each class.
    """
    cmap = plt.cm.get_cmap("gist_rainbow", num_classes)
    colors = [cmap(i, bytes=True)[:3][::-1] for i in range(num_classes)]
    return [tuple(int(v) for v in c) for c in colors]

def draw_box_title(
    color_mapping: List[Tuple[int]],
    class_names: List[str],
    box_thickness: int,
    image_np: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_id: int,
    pred_conf: float = None,
    is_target: bool = False,
):
    """
    Draw a rectangle with class name, confidence on the image
    :param color_mapping: A list of N RGB colors for each class
    :param class_names: A list of N class names
    :param box_thickness: Thickness of the bounding box (in pixels)
    :param image_np: Image in RGB format (H, W, C) where to draw the bounding box
    :param x1: X coordinate of the top left corner of the bounding box
    :param y1: Y coordinate of the top left corner of the bounding box
    :param x2: X coordinate of the bottom right corner of the bounding box
    :param y2: Y coordinate of the bottom right corner of the bounding box
    :param class_id: A corresponding class id
    :param pred_conf: Class confidence score (optional)
    :param is_target: Indicate if the bounding box is a ground-truth box or not

    """
    color = color_mapping[class_id]
    class_name = class_names[class_id]
    weight_estimate = estimate_vechicle_weight(class_name)

    if is_target:
        title = f"[GT] {class_name}"
    else:
        title = f'[Pred:{class_name}] [Conf:{str(round(pred_conf, 2)) if pred_conf is not None else ""}] [lbs:{weight_estimate}]'

    image_np = draw_bbox(image=image_np, title=title, x1=x1, y1=y1, x2=x2, y2=y2, box_thickness=box_thickness, color=color)
    return image_np
# def draw_bboxtil_mod(frame, results, class_names):
#     detections = results['output0'].numpy()[0]  # Convert to numpy and remove batch dimension\
#     color_mapping = generate_color_mapping(len(class_names))
#     image = frame.copy()
#     for detection in detections:
#         x_center, y_center, width, height, obj_score, *class_probs = detection[:5 + len(class_names)]
#         image = draw_box_title(image_np=frame,
#             x1=int(x_center),
#             y1=int(y_center),
#             x2=int(width),
#             y2=int(height),
#             class_id=np.argmax(class_probs),
#             class_names=class_names,
#             color_mapping=color_mapping,
#             box_thickness=2,
#             pred_conf=obj_score,
#         )
#
#     # fig = plt.figure(figsize=(8, 8))
#     # # plt.imshow(image)
#     # plt.tight_layout()
#     # # plt.show()
#     # # plt.imsave(f"C:/Users/Marc Hernandez/Documents/UCLA/ECE 202A/EVMEP/software/software_files/annotated_test_images/{name}",image_me)
#     # plt.close(fig)
#     cv2.imshow('Object Detection', image)


def draw_bbox(
    image: np.ndarray,
    title: Optional[str],
    color: Tuple[int, int, int],
    box_thickness: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Draw a bounding box on an image.

    :param image:           Image on which to draw the bounding box.
    :param color:           RGB values of the color of the bounding box.
    :param title:           Title to display inside the bounding box.
    :param box_thickness:   Thickness of the bounding box border.
    :param x1:              x-coordinate of the top-left corner of the bounding box.
    :param y1:              y-coordinate of the top-left corner of the bounding box.
    :param x2:              x-coordinate of the bottom-right corner of the bounding box.
    :param y2:              y-coordinate of the bottom-right corner of the bounding box.
    """

    overlay = image.copy()
    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

    if title is not None or title != "":
        # Adapt font size to image shape.
        # This is required because small images require small font size, but this makes the title look bad,
        # so when possible we increase the font size to a more appropriate value.
        font_size = 0.25 + 0.07 * min(overlay.shape[:2]) / 100
        font_size = max(font_size, 0.2)  # Set min font_size to 0.5
        font_size = min(font_size, 0.3)  # Set max font_size to 0.8

        overlay = draw_text_box(image=overlay, text=title, x=x1, y=y1, font=2, font_size=font_size, background_color=color, thickness=1)

    return cv2.addWeighted(overlay, 0.75, image, 0.25, 0)

def draw_text_box(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    font_size: float,
    background_color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Draw a text inside a box

    :param image:               The image on which to draw the text box.
    :param text:                The text to display in the text box.
    :param x:                   The x-coordinate of the top-left corner of the text box.
    :param y:                   The y-coordinate of the top-left corner of the text box.
    :param font:                The font to use for the text.
    :param font_size:           The size of the font to use.
    :param background_color:    The color of the text box and text as a tuple of three integers representing RGB values.
    :param thickness:           The thickness of the text.
    :return: Image with the text inside the box.
    """
    text_color = best_text_color(background_color)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, thickness)
    text_left_offset = 7

    image = cv2.rectangle(image, (x, y), (x + text_width + text_left_offset, y - text_height - int(15 * font_size)), background_color, -1)
    image = cv2.putText(image, text, (x + text_left_offset, y - int(10 * font_size)), font, font_size, text_color, thickness, lineType=cv2.LINE_AA)
    return image
def best_text_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Determine the best color for text to be visible on a given background color.

    :param background_color: RGB values of the background color.
    :return: RGB values of the best text color for the given background color.
    """

    # If the brightness is greater than 0.5, use black text; otherwise, use white text.
    if compute_brightness(background_color) > 0.5:
        return (0, 0, 0)  # Black
    else:
        return (255, 255, 255)  # White


def compute_brightness(color: Tuple[int, int, int]) -> float:
    """Computes the brightness of a given color in RGB format. From https://alienryderflex.com/hsp.html

    :param color: A tuple of three integers representing the RGB values of the color.
    :return: The brightness of the color.
    """
    return (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[0]) / 255

def show_predictions_from_batch_format(image, predictions):
    num_predictions, pred_boxes, pred_scores, pred_classes = predictions

    assert num_predictions.shape[0] == 1, "Only batch size of 1 is supported by this function"

    num_predictions = int(num_predictions.item())
    pred_boxes = pred_boxes[0, :num_predictions]
    pred_scores = pred_scores[0, :num_predictions]
    pred_classes = pred_classes[0, :num_predictions]

    image = image.copy()
    class_names = ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'estate',
               'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible']
    color_mapping = generate_color_mapping(len(class_names))

    for (x1, y1, x2, y2, class_score, class_index) in zip(pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3], pred_scores, pred_classes):
        image = draw_box_title(
            image_np=image,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            class_id=class_index,
            class_names=class_names,
            color_mapping=color_mapping,
            box_thickness=2,
            pred_conf=class_score,
        )

    plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    plt.tight_layout()
    #plt.show()
    plt.imsave(f"Z:/Test_Images/{name}", image)

def show_predictions_from_flat_format(image, predictions, name):
    [flat_predictions] = predictions

    image = image.copy()
    class_names = ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'wagon',
               'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible']
    color_mapping = generate_color_mapping(len(class_names))

    for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
        class_index = int(class_index)
        image = draw_box_title(
                    image_np=image,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=class_index,
                    class_names=class_names,
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=class_score,
                )

    fig = plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    plt.tight_layout()
    #plt.show()
    plt.imsave(f"C:/Users/Marc Hernandez/Documents/UCLA/ECE 202A/EVMEP/software/software_files/annotated_test_images/{name}", image)
    plt.close(fig)

def show_predictions_from_flat_format_no_save(image, predictions):
    [flat_predictions] = predictions

    image = image.copy()
    class_names = ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'wagon',
               'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible']
    color_mapping = generate_color_mapping(len(class_names))

    for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
        class_index = int(class_index)
        annotated = draw_box_title(
                    image_np=image,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=class_index,
                    class_names=class_names,
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=class_score,
                )

    # return annotated
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()
    # plt.imsave(f"Z:/Test_Images/{name}", image)