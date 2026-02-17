#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="./data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="RN50"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
# Use NICE-EEG preprocessed files directly (no copying/renaming required).
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="TSConv"
BATCH_SIZE=1024
LEARNING_RATE=1e-4
NUM_EPOCHS=50
# SELECTED_CHANNELS=('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
SELECTED_CHANNELS=() # "Oz" "O1" "O2" "POz" "PO3" "PO4" "PO7" "PO8" "Pz" "P1" "P2" "P3" "P4" "P5" "P6" "P7" "P8" "TP7" "TP8" "T7" "T8" "FT7" "FT8")
PROJECTOR="linear"
FEATURE_DIM=512
OUTPUT_DIR="./results/things_eeg/inter-subjects"
NUM_WORKERS="4" # "$(nproc)"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${OUTPUT_DIR}/last_run.txt"

for SUB_ID in {1..2}
do
    OUTPUT_NAME=$(printf "sub-%02d" $SUB_ID)
    echo "Training subject ${SUB_ID}..."

    TRAIN_IDS=""
    for i in {1..10}
    do
        if [ "$i" -ne "$SUB_ID" ]; then
            TRAIN_IDS+="$i "
        fi
    done

    python train.py \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids $TRAIN_IDS \
        --test_subject_ids $SUB_ID \
        --softplus \
        --num_epochs "$NUM_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE"  \
        --output_dir "$RUN_DIR" \
        --selected_channels "${SELECTED_CHANNELS[@]}" \
        --num_workers "$NUM_WORKERS" \
        --img_l2norm \
        --projector "$PROJECTOR" \
        --feature_dim "$FEATURE_DIM" \
        --data_average \
        --save_weights \
        --ivae \
        --z_s_dim 16 \
        --z_i_dim 256 \
        --z_is_dim 32 \
        --z_n_dim 8 \
        --beta_s 1.0 \
        --beta_i 1.0 \
        --beta_is 1.0 \
        --beta_n 1.0 \
        --gamma_cl 1.0 \
        --C_max 25.0 \
        --C_stop_iter 10000 \
        --ivae_hidden_dim 512 \
        --subj_emb_dim 32 \
        --ivae_n_layers 1 \
        --n_subjects 11 \
        --retrieval_feature "z_i" \
        --seed 2025;
done

python compute_avg_results.py --result_dir "$RUN_DIR";

        # --image_aug
        # --aug_image_feature_dirs "./data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
        # --eeg_aug
        # --eeg_aug_type "smooth"
        # --image_test_aug

        # ── iVAE flags (uncomment to enable) ──
        # --ivae
        # --z_s_dim 16
        # --z_i_dim 256
        # --z_is_dim 32
        # --z_n_dim 8
        # --beta_s 1.0
        # --beta_i 1.0
        # --beta_is 1.0
        # --beta_n 1.0
        # --gamma_cl 1.0
        # --C_max 25.0
        # --C_stop_iter 10000
        # --ivae_hidden_dim 512
        # --subj_emb_dim 32
        # --ivae_n_layers 1
        # --n_subjects 10
        # --retrieval_feature "z_i"      # or "full_z"
        # --reconstruct_raw_eeg          # decode to raw EEG instead of backbone embedding
        # --multi_positive_loss

        # ── Scheduler flags (uncomment to enable) ──
        # --scheduler
        # --milestones 20 35
        # --scheduler_gamma 0.1
        # --warmup_steps 500
        # --warmup_factor 0.333
        # --warmup_method "linear"
