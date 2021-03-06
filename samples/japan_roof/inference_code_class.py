import os
import sys
import random
import numpy as np
import tensorflow as tf
from skimage import io
import cv2
import colorsys
import time

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from samples.japan_roof import japan_roof

global graph


class InferenceConfig(japan_roof.PascalVOCConfig().__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceClass():
    def __init__(self, MODEL_DIR, is_gpu=False, test_mode="inference"):
        self.model = None
        self.show_bbox = True
        self.show_mask = True
        self.captions = None

        self.config = InferenceConfig()

        # ['BG', 'goodroof', 'parkinglot', 'road', 'trees', 'river', 'field', 'park', 'facility', 'solarpanel']
        self.config.NUM_CLASSES = 1 + 9
        self.DEVICE = None

        # /cpu:0 or /gpu:0
        if is_gpu:
            self.DEVICE = '/gpu:0'
        else:
            self.DEVICE = '/cpu:0'

        self.TEST_MODE = test_mode
        self.ckpt_number = '0422'
        self.MODEL_DIR = MODEL_DIR
        self.weights_path = os.path.join(self.MODEL_DIR, "mask_rcnn_pascalvoc_" + str(self.ckpt_number) + ".h5")
        self.result_save_dir = "./images/output/"

        self.model_init()

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image

    # 면적 계산을 위한 행렬의 합을 계산
    def solution(self, arr1, arr2):
        return sum([arr1, arr2])

    def random_colors(self, N, bright=True):
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

    def model_init(self):
        global graph
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR,
                                  config=self.config)
        print(self.weights_path)
        self.model.load_weights(self.weights_path, by_name=True)

        graph = tf.get_default_graph()


    def inference(self, image):
        with tf.device(self.DEVICE):
            with graph.as_default():
                loop_start_time = time.time()

                # png 이미지에 alpha 채널이 있다면 제거 (640, 640 ,4)  >> (640, 640, 3)
                if image.shape[-1] == 4:
                    image = image[..., :3]

                inference_start_time = time.time()
                results = self.model.detect([image], verbose=1)
                inference_end_time = time.time()

                # Display results
                r = results[0]

                boxes = r['rois']
                masks = r['masks']
                class_ids = r['class_ids']
                class_names = ['BG', 'goodroof', 'parkinglot', 'road', 'trees', 'river', 'field', 'park',
                               'facility', 'solarpanel']
                scores = r['scores']

                # Number of instances
                N = boxes.shape[0]
                print('number_of_instances : ', N)
                if not N:
                    print("\n*** No instances to display *** \n")
                else:
                    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

                # Generate random colors
                colors = self.random_colors(N)

                # Show area outside image boundaries.
                height, width = image.shape[:2]

                masked_image = image.astype(np.uint32).copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for i in range(N):
                    float_color = colors[i]
                    color = [int(element * 255) for element in float_color]

                    # Bounding box
                    if not np.any(boxes[i]):
                        # Skip this instance. Has no bbox. Likely lost in image cropping.
                        continue
                    y1, x1, y2, x2 = boxes[i]

                    if self.show_bbox:
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

                    # Label
                    if not self.captions:
                        class_id = class_ids[i]
                        score = scores[i] if scores is not None else None
                        label = class_names[class_id]
                        caption = "{} {:.3f}".format(label, score) if score else label
                    else:
                        caption = self.captions[i]
                    cv2.putText(image, caption, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Mask
                    mask = masks[:, :, i]
                    if self.show_mask:
                        masked_image = self.apply_mask(image, mask, color)
                        image = masked_image

                loop_end_time = time.time()
                total_elapsed_time = loop_end_time - loop_start_time
                inference_time = inference_end_time - inference_start_time
                print('total_elapsed_time : ', total_elapsed_time)
                print('inference_time : ', inference_time)

                cv2.imwrite(self.result_save_dir + 'input_image.png', image)

                return image, inference_time


if __name__ == '__main__':
    image = io.imread('./images/input/[0,0](135.555362E,34.640033N)_center_(135.55493E,34.640385N)min_(135.55579400000002E,34.639681N)_max_zoom_20_size_640x640.png')

    model_path = './models'
    obj = InferenceClass(model_path)

    obj.inference(image)


