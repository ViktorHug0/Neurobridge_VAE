#!/bin/bash
set -euo pipefail
trap 'echo "Script Error"' ERR

PYTHON_BIN="${PYTHON_BIN:-python3}"

IMAGE_FEATURE_BASE_DIR="${IMAGE_FEATURE_BASE_DIR:-./data/things_eeg/image_feature}"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-RN50}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
# Use NICE-EEG preprocessed files directly (no copying/renaming required).
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz}"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-EEGProject}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-35}"
# SELECTED_CHANNELS=('P7' 'P5' 'P3' 'P1' 'Pz' 'P2' 'P4' 'P6' 'P8' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'O1' 'Oz' 'O2')
SELECTED_CHANNELS=() # "Oz" "O1" "O2" "POz" "PO3" "PO4" "PO7" "PO8" "Pz" "P1" "P2" "P3" "P4" "P5" "P6" "P7" "P8" "TP7" "TP8" "T7" "T8" "FT7" "FT8")
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
NUM_WORKERS="${NUM_WORKERS:-4}" # "$(nproc)"
SUBJECT_PROBE_HOLDOUT="${SUBJECT_PROBE_HOLDOUT:-true}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-Neurobridge_VAE}"
WANDB_COLLECTION="${WANDB_COLLECTION:-runs}"

# iVAE/CL defaults (can be overridden via env for sweeps).
GAMMA_CL="${GAMMA_CL:-1.0}"
N_SUBJECTS="${N_SUBJECTS:-11}"
Z_S_DIM="${Z_S_DIM:-16}"
Z_IS_DIM="${Z_IS_DIM:-32}"
Z_I_DIM="${Z_I_DIM:-256}"
Z_N_DIM="${Z_N_DIM:-4}"
BETA_S="${BETA_S:-0.01}"
BETA_IS="${BETA_IS:-0.01}"
BETA_I="${BETA_I:-0.0001}"
BETA_N="${BETA_N:-0.01}"
LAMBDA_RECON="${LAMBDA_RECON:-0.1}"
LAMBDA_SUBJ_CLS="${LAMBDA_SUBJ_CLS:-0.0}"
LAMBDA_SUBJ_ADV="${LAMBDA_SUBJ_ADV:-0.0}"
GRL_LAMBDA="${GRL_LAMBDA:-0.0}"
C_MAX="${C_MAX:-10}"
C_STOP_ITER="${C_STOP_ITER:-10000}"
IVAE_HIDDEN_DIM="${IVAE_HIDDEN_DIM:-512}"
IVAE_N_LAYERS="${IVAE_N_LAYERS:-1}"
IMAGE_PRIOR_HIDDEN_DIM="${IMAGE_PRIOR_HIDDEN_DIM:-256}"
CL_COND_ON_SUBJECT="${CL_COND_ON_SUBJECT:-true}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}_VAE"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "${OUTPUT_DIR}/last_run.txt"

SUBJECT_PROBE_HOLDOUT_FLAG=""
if [ "$SUBJECT_PROBE_HOLDOUT" = true ]; then
    SUBJECT_PROBE_HOLDOUT_FLAG="--subject_probe_holdout"
fi

WANDB_FLAG=""
if [ "$WANDB" = true ]; then
    WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT} --wandb_collection ${WANDB_COLLECTION}"
fi

for SUB_ID in {1..10}
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

    CL_COND_FLAG="--no_cl_cond_on_subject"
    if [ "$CL_COND_ON_SUBJECT" = true ]; then
        CL_COND_FLAG="--cl_cond_on_subject"
    fi

    "${PYTHON_BIN}" train.py \
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
        --gamma_cl "$GAMMA_CL" \
        --n_subjects "$N_SUBJECTS" \
        --multi_positive_loss \
        $CL_COND_FLAG \
        $SUBJECT_PROBE_HOLDOUT_FLAG \
        --ivae \
        --z_s_dim "$Z_S_DIM" \
        --z_is_dim "$Z_IS_DIM" \
        --z_i_dim "$Z_I_DIM" \
        --z_n_dim "$Z_N_DIM" \
        --beta_s "$BETA_S" \
        --beta_is "$BETA_IS" \
        --beta_i "$BETA_I" \
        --beta_n "$BETA_N" \
        --lambda_recon "$LAMBDA_RECON" \
        --lambda_subj_cls "$LAMBDA_SUBJ_CLS" \
        --lambda_subj_adv "$LAMBDA_SUBJ_ADV" \
        --grl_lambda "$GRL_LAMBDA" \
        --C_max "$C_MAX" \
        --C_stop_iter "$C_STOP_ITER" \
        --ivae_hidden_dim "$IVAE_HIDDEN_DIM" \
        --ivae_n_layers "$IVAE_N_LAYERS" \
        --image_prior_hidden_dim "$IMAGE_PRIOR_HIDDEN_DIM" \
        $WANDB_FLAG ;
done

"${PYTHON_BIN}" compute_avg_results.py --result_dir "$RUN_DIR";
        # --image_aug    
        # --aug_image_feature_dirs "./data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
        # --eeg_aug
        # --eeg_aug_type "smooth"
        # --image_test_aug

        # ── iVAE flags (uncomment to enable) ──
        # --ivae
        # --z_s_dim 16
        # --z_is_dim 16
        # --z_i_dim 256
        # --z_n_dim 16
        # --beta_s 1.0
        # --beta_is 1.0
        # --beta_i 1.0
        # --beta_n 1.0
        # --gamma_cl 1.0
        # --lambda_subj_cls 1.0
        # --lambda_subj_adv 0.1
        # --grl_lambda 0.1
        # --ivae_hidden_dim 512
        # --ivae_n_layers 1
        # --n_subjects 10
        # --cl_cond_on_subject
        # --multi_positive_loss

        # ── Scheduler flags (uncomment to enable) ──
        # --scheduler
        # --milestones 20 35
        # --scheduler_gamma 0.1
        # --warmup_steps 500
        # --warmup_factor 0.333
        # --warmup_method "linear"

