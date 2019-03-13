@ECHO off

:: EDIT these paths to your directories, respectively:

:: MaskRCNN folder (delete the word "rem" from the beginning of the line and set the path if you already have
:: a MaskRCNN folder - in this case, move the word "rem" from between the beginning of the last 2 lines):
rem set "pathToInsert=D:\Mask_RCNN\Mask_RCNN-2.1"

:: working directory where you downloaded the code and will have the output under ~\kaggle_workflow\outputs\maskrcnn:
set root_dir=%~dp0

:: directory of your masks for style transfer learning input. Provide ONLY if you have custom masks for your input
:: images, otherwise the default folder will be the presegmented masks generated by our pipeline.
rem set images_dir="maskFolder"
:: -----------------------------------------------------------------------------


:: --- DO NOT EDIT from here ---
rem run_workflow_prepareStyle4singleExperiment.bat %root_dir% %images_dir%
run_workflow_prepareStyle4singleExperiment.bat %root_dir%