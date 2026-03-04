#!/bin/bash
set -euo pipefail
trap 'echo "Script Error"' ERR

# ---------------------------------------------------------------------------
# Sweep one parameter for a single held-out test subject.
# Runs train.py multiple times and writes a compact comparison CSV.
# ---------------------------------------------------------------------------

# =========================
# Base experiment settings
# =========================
IMAGE_FEATURE_BASE_DIR="./data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="RN50"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"

DEVICE="cuda:0"
EEG_ENCODER_TYPE="TSConv"
BATCH_SIZE=1024
LEARNING_RATE=1e-4
NUM_EPOCHS=40
SELECTED_CHANNELS=()
PROJECTOR="linear"
FEATURE_DIM=512
NUM_WORKERS="4"

# =========================
# Sweep settings
# =========================
TEST_SUBJECT=1
SWEEP_PARAM="grl_lambda"
SWEEP_VALUES=("0.0" "0.1" "0.5" "1.0" "5.0" "10")

# Optional: repeat each setting multiple times with different seeds.
N_REPEATS=1
BASE_SEED=2025

# =========================
# Fixed model settings
# =========================
Z_S_DIM=32
Z_IS_DIM=16
Z_I_DIM=512
Z_N_DIM=16
BETA_S=0.1
BETA_IS=0.1
BETA_I=0.0
BETA_N=0.1
LAMBDA_RECON=1.0
LAMBDA_SUBJ_CLS=0.0
LAMBDA_SUBJ_ADV=0.1
GRL_LAMBDA=1.0
C_MAX=5.0
C_STOP_ITER=10000
IVAE_HIDDEN_DIM=512
IVAE_N_LAYERS=1
N_SUBJECTS=11
GAMMA_CL=10.0
CL_COND_ON_SUBJECT=true

OUTPUT_DIR="./results/things_eeg/inter-subject-sweeps"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}-sub-$(printf "%02d" "${TEST_SUBJECT}")-${SWEEP_PARAM}"
mkdir -p "${RUN_DIR}"
echo "${RUN_DIR}" > "${OUTPUT_DIR}/last_run.txt"

echo "Sweep run dir: ${RUN_DIR}"
echo "Test subject: ${TEST_SUBJECT}"
echo "Sweep param:  ${SWEEP_PARAM}"
echo "Values:       ${SWEEP_VALUES[*]}"

# Build train subject list = all subjects except TEST_SUBJECT
TRAIN_IDS=""
for sid in {1..10}; do
    if [ "${sid}" -ne "${TEST_SUBJECT}" ]; then
        TRAIN_IDS+="${sid} "
    fi
done

for value in "${SWEEP_VALUES[@]}"; do
    for ((rep = 0; rep < N_REPEATS; rep++)); do
        seed=$((BASE_SEED + rep))
        value_tag="${value//./p}"
        output_name="$(printf "sub-%02d-%s-%s-r%d" "${TEST_SUBJECT}" "${SWEEP_PARAM}" "${value_tag}" "${rep}")"

        echo ""
        echo ">>> Running ${SWEEP_PARAM}=${value} (repeat=${rep}, seed=${seed})"

        CL_COND_FLAG="--no_cl_cond_on_subject"
        if [ "${CL_COND_ON_SUBJECT}" = true ]; then
            CL_COND_FLAG="--cl_cond_on_subject"
        fi

        python train.py \
            --batch_size "${BATCH_SIZE}" \
            --learning_rate "${LEARNING_RATE}" \
            --output_name "${output_name}" \
            --eeg_encoder_type "${EEG_ENCODER_TYPE}" \
            --train_subject_ids ${TRAIN_IDS} \
            --test_subject_ids "${TEST_SUBJECT}" \
            --softplus \
            --num_epochs "${NUM_EPOCHS}" \
            --image_feature_dir "${IMAGE_FEATURE_DIR}" \
            --text_feature_dir "${TEXT_FEATURE_DIR}" \
            --eeg_data_dir "${EEG_DATA_DIR}" \
            --device "${DEVICE}" \
            --output_dir "${RUN_DIR}" \
            --selected_channels "${SELECTED_CHANNELS[@]}" \
            --num_workers "${NUM_WORKERS}" \
            --img_l2norm \
            --projector "${PROJECTOR}" \
            --feature_dim "${FEATURE_DIM}" \
            --data_average \
            --save_weights \
            --ivae \
            --z_s_dim "${Z_S_DIM}" \
            --z_is_dim "${Z_IS_DIM}" \
            --z_i_dim "${Z_I_DIM}" \
            --z_n_dim "${Z_N_DIM}" \
            --beta_s "${BETA_S}" \
            --beta_is "${BETA_IS}" \
            --beta_i "${BETA_I}" \
            --beta_n "${BETA_N}" \
            --lambda_recon "${LAMBDA_RECON}" \
            --lambda_subj_cls "${LAMBDA_SUBJ_CLS}" \
            --lambda_subj_adv "${LAMBDA_SUBJ_ADV}" \
            --grl_lambda "${GRL_LAMBDA}" \
            --C_max "${C_MAX}" \
            --C_stop_iter "${C_STOP_ITER}" \
            --ivae_hidden_dim "${IVAE_HIDDEN_DIM}" \
            --ivae_n_layers "${IVAE_N_LAYERS}" \
            --n_subjects "${N_SUBJECTS}" \
            ${CL_COND_FLAG} \
            --seed "${seed}" \
            --gamma_cl "${GAMMA_CL}" \
            --"${SWEEP_PARAM}" "${value}"
    done
done

# Build a compact comparison CSV (one row per sweep run)
python3 - <<'PY' "${RUN_DIR}" "${SWEEP_PARAM}"
import os
import sys
import re
import pandas as pd

run_dir = sys.argv[1]
sweep_param = sys.argv[2]
rows = []

for d in sorted(os.listdir(run_dir)):
    full = os.path.join(run_dir, d)
    if not os.path.isdir(full):
        continue
    result_csv = os.path.join(full, "result.csv")
    cfg_json = os.path.join(full, "train_config.json")
    if not (os.path.exists(result_csv) and os.path.exists(cfg_json)):
        continue

    res = pd.read_csv(result_csv).iloc[0].to_dict()
    cfg = pd.read_json(cfg_json, typ="series")
    row = {
        "run": d,
        sweep_param: cfg.get(sweep_param, None),
        "seed": cfg.get("seed", None),
        "top1 acc": float(res.get("top1 acc", float("nan"))),
        "top5 acc": float(res.get("top5 acc", float("nan"))),
        "best top1 acc": float(res.get("best top1 acc", float("nan"))),
        "best top5 acc": float(res.get("best top5 acc", float("nan"))),
        "best test contrastive loss": float(res.get("best test contrastive loss", float("nan"))),
        "best test loss": float(res.get("best test loss", float("nan"))),
        "best epoch": int(res.get("best epoch", -1)),
    }
    rows.append(row)

if not rows:
    raise FileNotFoundError(f"No completed result rows found in {run_dir}")

df = pd.DataFrame(rows)
df = df.sort_values(by=[sweep_param, "seed"], kind="stable")
out_csv = os.path.join(run_dir, "sweep_results.csv")
df.to_csv(out_csv, index=False)
print(df)
print(f"\nSaved: {out_csv}")
PY

echo ""
echo "Done. Inspect:"
echo "  ${RUN_DIR}/sweep_results.csv"
