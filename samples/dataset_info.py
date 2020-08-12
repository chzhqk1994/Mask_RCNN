class Config(object):
    NAME = 'pascalvoc'

    # This will make sub-directory under the default-log directory automatically by this name
    DATASET_DESCRIPTION = 'no_trees'
    MAX_IMAGE_WIDTH = 640
    MAX_IMAGE_HEIGHT = 640

    CLASS_NAMES = ['goodroof', 'parkinglot', 'road', 'trees', 'river', 'field', 'park', 'facility', 'solarpanel']

    TRAINING_STEP = 2000000
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8

    GPU_NUMBER = 2
