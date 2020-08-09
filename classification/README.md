# TensorFlow2.2_Image_Classification
## Usage
usage: demo_classification.py [-h] [--train] [--test] [--predict]
                              [--rescale RESCALE] [--shear_range SHEAR_RANGE]
                              [--zoom_range ZOOM_RANGE]
                              [--rotation_range ROTATION_RANGE]
                              [--horizontal_flip HORIZONTAL_FLIP]
                              [--target_size TARGET_SIZE]
                              [--class_mode CLASS_MODE] [--lr LR]
                              [--bsize BSIZE] [--maxepoch MAXEPOCH]
                              [--train_dir TRAIN_DIR] [--valid_dir VALID_DIR]
                              [--test_dir TEST_DIR] [--log_path LOG_PATH]
                              [--save_path SAVE_PATH]
## command for training       
python ./examples/demo_classification --train [--bsize BSIZE] [--maxepoch MAXEPOCH]...
## test
python ./examples/demo_classification --test [--test_dir TEST_DIR]
## predict
python ./examples/demo_classification --predict <image_dir>