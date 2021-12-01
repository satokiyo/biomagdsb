#!/bin/bash

# EDIT these paths to your directories, respectively:

# MaskRCNN folder (delete the word "#" from the beginning of the line and set the path if you already have
# a MaskRCNN folder - in this case, move the word "#" from between the beginning of the last 2 lines):
PATHTOINSERT="/media/HDD2/20211026_hr_kaggle/biomagdsb/Mask_RCNN/mrcnn"


ROOT_DIR="."

# directory of your images to segment:
IMAGES_DIR="testImages"
# -----------------------------------------------------------------------------


# --- DO NOT EDIT from here ---
source run_workflow_predictOnly_full.sh $ROOT_DIR $IMAGES_DIR $PATHTOINSERT
#source run_workflow_predictOnly_full.sh $ROOT_DIR $IMAGES_DIR