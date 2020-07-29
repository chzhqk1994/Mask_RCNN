import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
from skimage import io
import cv2
import colorsys

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log

from samples.japan_roof import japan_roof
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ckpt_number = '0422'
MODEL_DIR = "./models/"
weights_path = os.path.join(MODEL_DIR, "mask_rcnn_pascalvoc_" + str(ckpt_number) + ".h5")

result_save_dir = "./images/output/"

config = japan_roof.PascalVOCConfig()


# class InferenceConfig(config.__class__):
#     # Run detection on one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1


# config = InferenceConfig()

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

# 면적 계산을 위한 행렬의 합을 계산

def solution(arr1, arr2):
    return [[c + d for c, d in zip(a, b)] for a, b in zip(arr1,arr2)]


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

    model.load_weights(weights_path, by_name=True)

    for path, dirs, files in os.walk("./images/input/"):
        # for path, dirs, files in os.walk("/root/japan_roof_with_trees/"):
        for file in files:
            loop_start_time = time.time()

            image_path = os.path.join(path, file)
            image = io.imread(image_path)

            # png 이미지에 alpha 채널이 있다면 제거 (640, 640 ,4)  >> (640, 640, 3)
            if image.shape[-1] == 4:
                image = image[..., :3]

            inference_start_time = time.time()
            results = model.detect([image], verbose=1)
            inference_end_time = time.time()

            # Display results
            r = results[0]

            boxes = r['rois']
            masks = r['masks']
            class_ids = r['class_ids']
            class_names = ['BG', 'goodroof', 'parkinglot', 'road', 'trees', 'river', 'field', 'park', 'facility', 'solarpanel']
            scores = r['scores']
            show_bbox = True
            show_mask = True
            captions = None

            # Number of instances
            N = boxes.shape[0]
            print('number_of_instances : ', N)
            if not N:
                print("\n*** No instances to display *** \n")
            else:
                assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

            # Generate random colors
            colors = random_colors(N)

            # Show area outside image boundaries.
            height, width = image.shape[:2]

            masked_image = image.astype(np.uint32).copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 이미지 사이즈의 Flase array 준비
            flat_mask = np.full((640, 640), False, dtype=bool)
            trees_mask = np.full((640, 640), False, dtype=bool)
            solarpanel_mask = np.full((640, 640), False, dtype=bool)

            for i in range(N):
                float_color = colors[i]
                color = [int(element * 255) for element in float_color]

                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]

                if show_bbox:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

                # Label
                if not captions:
                    class_id = class_ids[i]
                    score = scores[i] if scores is not None else None
                    label = class_names[class_id]
                    caption = "{} {:.3f}".format(label, score) if score else label
                else:
                    caption = captions[i]
                cv2.putText(image, caption, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Mask
                mask = masks[:, :, i]
                if show_mask:
                    masked_image = apply_mask(image, mask, color)

                start_mask_time = time.time()
                if label == 'goodroof':
                    flat_mask = solution(flat_mask, mask)

                if label == 'trees':
                    trees_mask = solution(trees_mask, mask)

                if label == 'solarpanel':
                    solarpanel_mask = solution(solarpanel_mask, mask)

                end_mask_time = time.time()

                #             print("label : ", label)
                #             print("instance mask size : ", np.count_nonzero(mask))
                #             print("flat_mask : ", np.count_nonzero(flat_mask))

                # 한 이미지의 마지막 루프 때 면적 계산결과를 표시
                start_draw_cv_time = time.time()
                if i == N - 1:
                    flat_text = "Ratio of Flat : " + str(
                        "%0.2f" % (np.count_nonzero(flat_mask) / 409600))  # 640*640 = 409600
                    solarpanel_text = "Ratio of solarpanel : " + str(
                        "%0.2f" % (np.count_nonzero(solarpanel_mask) / 409600))  # 640*640 = 409600
                    trees_text = "Ratio of trees : " + str(
                        "%0.2f" % (np.count_nonzero(trees_mask) / 409600))  # 640*640 = 409600

                    cv2.rectangle(masked_image, (380, 580), (640, 640), (255, 255, 255), (-1))
                    cv2.putText(masked_image, flat_text, (380, 580 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(masked_image, solarpanel_text, (380, 580 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 2)
                    cv2.putText(masked_image, trees_text, (380, 580 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                end_draw_cv_time = time.time()

            loop_end_time = time.time()
            print("Inference time : ", inference_end_time - inference_start_time)
            print("Mask time : ", end_mask_time - start_mask_time)
            print("Draw OpenCV time : ", end_draw_cv_time - start_draw_cv_time)
            print("Total Elapsed time : ", loop_end_time - loop_start_time)
            print("\n\n")
            # cv2.imwrite(result_save_dir + file, masked_image)
            # print("Image count : ", cnt)
