import os
import sys
sys.path.insert(1, "../../lib")

DIR_PATH__ = os.path.dirname(os.path.realpath(__file__))
DATA_2D_PATH__ = os.path.join(DIR_PATH__, "..", "data", "datasets", "output_current_model_sphere")
DATA_3D_PATH__ = os.path.join(DIR_PATH__, "..", "data", "datasets", "output_current_3d_model_sphere")

DATA_TR_2D_PATH__ = os.path.join(DATA_2D_PATH__, "tr")
DATA_TE_2D_PATH__ = os.path.join(DATA_2D_PATH__, "te")
DATA_TR_3D_PATH__ = os.path.join(DATA_3D_PATH__, "tr")
# _DATA_TE_3D_PATH = os.path.join(DATA_3D_PATH__, "te") # Currently 3D has no test datasets

NUM_IMAGES_TR_2D__ = len(os.listdir(DATA_TR_2D_PATH__))
NUM_IMAGES_TE_2D__ = len(os.listdir(DATA_TE_2D_PATH__))
NUM_IMAGES_TR_3D__ = len(os.listdir(DATA_TR_3D_PATH__))
# _NUM_IMAGES_TE_3D = len(os.listdir(_DATA_TE_3D_PATH))

# Assume each images has the same number of frames
NUM_FRAMES_TR_2D__ = 64
NUM_FRAMES_TE_2D__ = 64
NUM_FRAMES_TR_3D__ = 64

FRAME_INCREMENT__ = 4
