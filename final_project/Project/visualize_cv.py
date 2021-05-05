#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import cv2
import numpy as np
import os
import sys


# In[2]:


ROOT_DIR = '../../aktwelve_Mask_RCNN'
assert os.path.exists(ROOT_DIR)
sys.path.append(ROOT_DIR) 
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn.config import Config

# In[14]:



MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[24]:
class MallConfig(Config):
    """Configuration for training on the classroom_data dataset.
    Derives from the base Config class and overrides values specific
    to the classroom_data dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Mall"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    LEARNING_RATE = 0.002

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + person

    # Change this later based on the dimension of images formed from extracting from video
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 200

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 4

    BACKBONE = 'resnet101'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000


config = MallConfig()
config.display()

class InferenceConfig(MallConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85

inference_config = InferenceConfig()
inference_config.display()


# In[25]:


model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# In[26]:


model.load_weights('../../aktwelve_Mask_RCNN/logs/mall20210423T1210/mask_rcnn_mall_0003.h5', by_name=True)


# In[27]:


class_names = [
    'person'
]


# In[28]:


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


# In[29]:


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


# In[30]:


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]-1]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


# In[33]:


# if __name__ == '__main__':
#     capture = cv2.VideoCapture('..\\..\\UBI_FIGHTS\\videos\\fight\\F_8_1_0_0_0')
#
#         # these 2 lines can be removed if you dont have a 1080p camera.
#     #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#
#     while True:
#         ret, frame = capture.read()
#         results = model.detect([frame], verbose=0)
#         r = results[0]
#         frame = display_instances(
#             frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
#         )
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     capture.release()
#     cv2.destroyAllWindows()


# In[ ]:




