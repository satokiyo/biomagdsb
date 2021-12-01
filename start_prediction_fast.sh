#!/bin/bash

# EDIT these paths to your directories, respectively:

# MaskRCNN folder (delete the word "#" from the beginning of the line and set the path if you already have
# a MaskRCNN folder - in this case, move the word "#" from between the beginning of the last 2 lines):
PATHTOINSERT="/media/HDD2/20211026_hr_kaggle/biomagdsb/Mask_RCNN/mrcnn"

# working directory where you downloaded the code and will have the output under ~\kaggle_workflow\outputs\postprocessing:
ROOT_DIR="."

# directory of your images to segment:
#IMAGES_DIR="testImages/kaggle2018_test"
IMAGES_DIR="/media/HDD2/20211104_panet/sample_data/ROI/cytology_sample_x20_512x512/train/sample_data/coco"
# -----------------------------------------------------------------------------


# --- DO NOT EDIT from here ---
source run_workflow_predictOnly_fast.sh $ROOT_DIR $IMAGES_DIR $PATHTOINSERT
#source run_workflow_predictOnly_fast.sh $ROOT_DIR $IMAGES_DIR