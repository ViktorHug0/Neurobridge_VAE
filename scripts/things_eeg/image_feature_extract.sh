#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

# Point to your existing image set (from NICE-EEG) so this script works without copying data.
IMAGE_SET_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"

python extract_feature.py --image_set_dir "$IMAGE_SET_DIR" --output_dir "./data/things_eeg/image_feature/RN50" --aug_type "None" --device "cuda:0"

for AUG_TYPE in GaussianBlur GaussianNoise LowResolution Mosaic
do
    python extract_feature.py --image_set_dir "$IMAGE_SET_DIR" --output_dir "./data/things_eeg/image_feature/RN50/${AUG_TYPE}" --aug_type "$AUG_TYPE" --device "cuda:0"
done

python fuse_feature.py --image_feature_dir "./data/things_eeg/image_feature/RN50" --aug_type "GaussianBlur" "GaussianNoise" "LowResolution" "Mosaic"