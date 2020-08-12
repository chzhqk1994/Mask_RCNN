"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 train.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 train.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 train.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 train.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 train.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import math
import platform
from samples.dataset_info import Config as DatasetConfig

os.environ['CUDA_VISIBLE_DEVICES'] = str(DatasetConfig.GPU_NUMBER)

FLATFORM = platform.system()

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from custom_lib import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
SUB_LOGS_DIR = os.path.join(DEFAULT_LOGS_DIR, DatasetConfig.DATASET_DESCRIPTION)

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.mkdir(DEFAULT_LOGS_DIR)

if not os.path.exists(SUB_LOGS_DIR):
    os.mkdir(SUB_LOGS_DIR)

############################################################
#  Configurations
############################################################


class MaskRcnnConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = DatasetConfig.NAME

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(DatasetConfig.CLASS_NAMES)  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = DatasetConfig.STEPS_PER_EPOCH

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = DatasetConfig.DETECTION_MIN_CONFIDENCE


class _InferenceConfig(MaskRcnnConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class MaskRcnnDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add
        for i in range(len(DatasetConfig.CLASS_NAMES)):
            self.add_class(DatasetConfig.NAME, i + 1, DatasetConfig.CLASS_NAMES[i])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        json_path = os.path.join(dataset_dir, 'json')
        image_path = os.path.join(dataset_dir, 'image')

        for path, dirs, files in os.walk(json_path):
            for file in files:
                if FLATFORM == 'Darwin':
                    if file == '.DS_Store':
                        continue

                num_ids = []
                json_file = os.path.join(path, file)
                annotations = json.load(open(json_file, 'r'))

                polygons = annotations['shapes']
                for i in polygons:
                    label = i["label"]
                    try:
                        # num_dis : 0 => background
                        num_ids.append(DatasetConfig.CLASS_NAMES.index(label) + 1)

                    except:
                        pass

                image_file_path = os.path.join(image_path, annotations['imagePath'])
                image = skimage.io.imread(image_file_path)
                height, width = image.shape[:2]

                print(json_file, image_file_path)

                self.add_image(
                    DatasetConfig.NAME,
                    image_id=annotations['imagePath'],  # use file name as a unique image id
                    path=image_file_path,
                    width=width, height=height,
                    polygons=polygons,
                    num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != DatasetConfig.NAME:
            return super(self.__class__, self).load_mask(image_id)

        num_ids = image_info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            is_error = False

            # Get indexes of pixels inside the polygon and set them to 1
            all_points_x = []
            all_points_y = []
            for point in p["points"]:
                if point[0] > DatasetConfig.MAX_IMAGE_WIDTH:
                    point[0] = DatasetConfig.MAX_IMAGE_WIDTH
                if point[1] > DatasetConfig.MAX_IMAGE_HEIGHT:
                    point[1] = DatasetConfig.MAX_IMAGE_HEIGHT
                all_points_x.append(point[0])
                all_points_y.append(point[1])

            # If shape_type is 'Circle'
            center_coord = []
            second_coord = []
            try:
                if p["shape_type"] == 'circle':
                    center_coord = [all_points_x[0], all_points_y[0]]
                    second_coord = [all_points_x[1], all_points_y[1]]

                    points_x, points_y = self.get_circle_coords(center_coord=center_coord, second_coord=second_coord,
                                                                num_points=20,
                                                                flag='separate')

                    for index in range(0, len(points_x)):
                        if points_x[index] > DatasetConfig.MAX_IMAGE_WIDTH:
                            points_x[index] = DatasetConfig.MAX_IMAGE_WIDTH
                        if points_y[index] > DatasetConfig.MAX_IMAGE_HEIGHT:
                            points_y[index] = DatasetConfig.MAX_IMAGE_HEIGHT
                        if points_x[index] < 0:
                            points_x[index] = 0
                        if points_y[index] < 0:
                            points_y[index] = 0

                    all_points_x = points_x
                    all_points_y = points_y

            except IndexError as e:
                is_error = True

                with open('error_file_circle.txt', 'a') as fd:
                    fd.write('image id : ' + str(image_info['id']) + '\n')
                    fd.write('shape type : ' + str(p['shape_type']) + '\n')
                    fd.write("center coord : " + str(center_coord) + '\n')
                    fd.write("second coord : " + str(second_coord) + '\n')
                    fd.write("x_coords : " + str(all_points_x) + '\n')
                    fd.write("y_coords : " + str(all_points_y) + '\n')
                    fd.write('\n')
                    fd.write('\n')

            # If shape_type is 'Rectangle'
            if p['shape_type'] == 'rectangle':
                all_points_x = [all_points_x[0], all_points_x[0], all_points_x[1], all_points_x[1]]
                all_points_y = [all_points_y[0], all_points_y[1], all_points_y[1], all_points_y[0]]

            if is_error:
                continue

            try:
                rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
                mask[rr, cc, i] = 1
            except IndexError as e:

                with open('error_file.txt', 'a') as fd:
                    fd.write('shape type : ' + str(p['shape_type']) + '\n')
                    fd.write("center coord : " + str(center_coord) + '\n')
                    fd.write("second coord : " + str(second_coord) + '\n')
                    fd.write("x_coords : " + str(all_points_x) + '\n')
                    fd.write("y_coords : " + str(all_points_y) + '\n')
                    fd.write('\n')
                    fd.write('\n')

        num_ids = np.array(num_ids, dtype=np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == DatasetConfig.NAME:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def rotate(self, x, y, r):
        rx = (x * math.cos(r)) - (y * math.sin(r))
        ry = (y * math.cos(r)) + (x * math.sin(r))
        return rx, ry

    def get_circle_coords(self, center_coord, second_coord, num_points=20, flag='pack'):
        # calculate distance_between_points
        x = center_coord[0] - second_coord[0]
        y = center_coord[1] - second_coord[1]
        radius = math.sqrt(pow(x, 2) + pow(y, 2))

        arc = (2 * math.pi) / num_points  # what is the angle between two of the points

        if flag == 'separate':
            points_x = []
            points_y = []
            for p in range(num_points):
                px, py = self.rotate(0, radius, arc * p)
                px += center_coord[0]
                py += center_coord[1]

                points_x.append(px)
                points_y.append(py)
            return points_x, points_y

        elif flag == 'pack':
            points = []
            for p in range(num_points):
                px, py = self.rotate(0, radius, arc * p)
                px += center_coord[0]
                py += center_coord[1]
                points.append([px, py])

            return points

        else:
            raise Exception("get_circle_coords FLAG Error : Only input 'pack' or 'separate'")


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MaskRcnnDataset()
    dataset_train.load_dataset(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MaskRcnnDataset()
    dataset_val.load_dataset(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    infconfig = _InferenceConfig()
    model_inference = modellib.MaskRCNN(mode="inference", config=infconfig, model_dir=SUB_LOGS_DIR)
    mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model, model_inference,
                                                                            dataset_val, DatasetConfig.CLASS_NAMES, calculate_at_every_X_epoch=4,
                                                                            verbose=1)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=DatasetConfig.TRAINING_STEP,
                layers='heads',
                custom_callbacks=[mean_average_precision_callback])


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect from dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "train":
        config = MaskRcnnConfig()
    else:
        class InferenceConfig(MaskRcnnConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=SUB_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=SUB_LOGS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
