#!/bin/bash
set -euo pipefail

# Run 20 iVAE experiments sequentially (each across all 10 subjects),
# and append per-experiment mean metrics to all_results.csv immediately.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_SCRIPT="./scripts/things_eeg/inter-subjects.sh"
OUTPUT_ROOT="./results/things_eeg/inter-subjects-20exp"
ALL_RESULTS_FILE="${OUTPUT_ROOT}/all_results.csv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${OUTPUT_ROOT}"

if [ ! -f "${ALL_RESULTS_FILE}" ]; then
    echo "timestamp,exp_id,run_id,status,gamma_cl,z_s_dim,z_is_dim,z_i_dim,z_n_dim,beta_s,beta_is,beta_i,beta_n,lambda_recon,lambda_subj_cls,lambda_subj_adv,grl_lambda,C_max,C_stop_iter,ivae_hidden_dim,ivae_n_layers,image_prior_hidden_dim,cl_cond_on_subject,mean_top1_acc,mean_top5_acc,mean_best_top1_acc,mean_best_top5_acc,mean_best_test_cl,mean_best_test_loss,mean_best_epoch,run_dir" > "${ALL_RESULTS_FILE}"
fi

Z_IS_DIM="${Z_IS_DIM:-16}"
Z_N_DIM="${Z_N_DIM:-16}"
BETA_IS="${BETA_IS:-0.1}"
BETA_N="${BETA_N:-0.1}"
CL_COND_ON_SUBJECT="${CL_COND_ON_SUBJECT:-true}"

EXPERIMENTS=(
    "CL-focus-baseline|2.0|16|256|0.01|0.00|0.10|0.00|0.00|0.00|0|10000|512|1|256|z_i"
    "CL-plus-recon|2.2|16|256|0.01|0.00|0.20|0.00|0.00|0.00|0|10000|512|1|256|z_i"
    "CL-plus-beta_s|2.5|16|256|0.20|0.00|0.10|0.00|0.00|0.00|0|10000|512|1|256|z_i"
    "CL-beta_s-and-recon|2.8|16|256|0.30|0.00|0.20|0.00|0.00|0.00|0|10000|512|1|256|z_i"
    "CL-subj-zs-light|2.5|16|256|0.10|0.00|0.10|0.05|0.00|0.00|0|10000|512|1|256|z_i"
    "CL-adversarial-light|2.5|16|256|0.10|0.00|0.10|0.00|0.05|0.10|0|10000|512|1|256|z_i"
    "CL-capacity-warmup|3.0|16|256|0.10|0.00|0.10|0.00|0.05|0.10|1.0|15000|512|1|256|z_i"
    "CL-capacity-high|3.0|16|256|0.15|0.05|0.10|0.00|0.10|0.20|2.0|20000|512|1|256|z_i"
    "CL-large-z_i|3.2|16|512|0.10|0.00|0.10|0.00|0.05|0.10|1.0|15000|512|1|256|z_i"
    "CL-large-z_i-deeper|3.5|16|512|0.20|0.00|0.10|0.05|0.10|0.20|2.0|20000|768|2|384|z_i"
    "CL-balanced-medium|4.0|32|384|0.20|0.05|0.20|0.10|0.10|0.30|2.0|20000|768|2|384|z_i"
    "CL-strong-adv|4.5|32|384|0.20|0.05|0.20|0.10|0.20|0.50|2.5|20000|768|2|384|z_i"
    "CL-strong-capacity|5.0|32|512|0.20|0.10|0.25|0.10|0.20|0.50|3.0|25000|768|2|384|z_i"
    "CL-zs-regularized|5.0|64|384|0.40|0.05|0.20|0.25|0.10|0.20|2.0|20000|768|2|384|z_i"
    "CL-zi-wide|5.5|32|640|0.20|0.10|0.20|0.10|0.20|0.50|3.0|25000|1024|2|512|z_i"
    "CL-max-focus|6.0|32|640|0.20|0.10|0.20|0.10|0.20|0.50|3.0|25000|1024|2|512|z_i"
    "CL-full-z-probe|6.0|32|512|0.20|0.10|0.20|0.10|0.20|0.50|2.0|20000|768|2|384|full_z"
    "CL-high-zs-full-z|6.5|64|512|0.50|0.10|0.25|0.30|0.20|0.50|3.0|25000|1024|2|512|full_z"
    "CL-deep-prior|7.0|64|640|0.40|0.10|0.25|0.20|0.20|0.50|3.0|30000|1024|3|512|z_i"
    "CL-extreme-dominant|8.0|64|768|0.50|0.20|0.30|0.20|0.20|0.80|3.5|30000|1024|3|512|z_i"
)

echo "Starting 20 experiments. Results will be appended to: ${ALL_RESULTS_FILE}"

# Resume support:
# Skip experiments that already have a successful (status=OK) row in all_results.csv.
declare -A DONE_OK
while IFS=',' read -r timestamp exp_id _run_id status _rest; do
    if [ "${timestamp}" = "timestamp" ] || [ -z "${exp_id}" ]; then
        continue
    fi
    if [ "${status}" = "OK" ]; then
        DONE_OK["${exp_id}"]=1
    fi
done < "${ALL_RESULTS_FILE}"

for idx in "${!EXPERIMENTS[@]}"; do
    exp_num=$((idx + 1))
    IFS='|' read -r EXP_ID GAMMA_CL Z_S_DIM Z_I_DIM BETA_S BETA_I LAMBDA_RECON LAMBDA_SUBJ_CLS LAMBDA_SUBJ_ADV GRL_LAMBDA C_MAX C_STOP_ITER IVAE_HIDDEN_DIM IVAE_N_LAYERS IMAGE_PRIOR_HIDDEN_DIM _unused_retrieval <<< "${EXPERIMENTS[$idx]}"

    if [[ -n "${DONE_OK[${EXP_ID}]:-}" ]]; then
        echo "[$(date +%H:%M:%S)] Experiment ${exp_num}/20: ${EXP_ID} -> already OK in all_results.csv, skipping."
        continue
    fi

    RUN_ID="$(date +%Y%m%d-%H%M%S)-exp$(printf '%02d' "${exp_num}")"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}_VAE"
    NOW="$(date +%Y-%m-%dT%H:%M:%S)"

    echo ""
    echo "[$(date +%H:%M:%S)] Experiment ${exp_num}/20: ${EXP_ID}"
    echo "  run_id=${RUN_ID}"
    echo "  gamma_cl=${GAMMA_CL} (must dominate)"

    dominance_ok="$(python3 - <<'PY' "${GAMMA_CL}" "${BETA_S}" "${BETA_IS}" "${BETA_I}" "${BETA_N}" "${LAMBDA_RECON}" "${LAMBDA_SUBJ_CLS}" "${LAMBDA_SUBJ_ADV}" "${GRL_LAMBDA}" "${C_MAX}"
import sys
vals = [float(x) for x in sys.argv[1:]]
gamma = vals[0]
others = vals[1:]
print("1" if gamma > max(others) else "0")
PY
)"
    if [ "${dominance_ok}" != "1" ]; then
        echo "${NOW},${EXP_ID},${RUN_ID},FAILED(gamma_not_dominant),${GAMMA_CL},${Z_S_DIM},${Z_IS_DIM},${Z_I_DIM},${Z_N_DIM},${BETA_S},${BETA_IS},${BETA_I},${BETA_N},${LAMBDA_RECON},${LAMBDA_SUBJ_CLS},${LAMBDA_SUBJ_ADV},${GRL_LAMBDA},${C_MAX},${C_STOP_ITER},${IVAE_HIDDEN_DIM},${IVAE_N_LAYERS},${IMAGE_PRIOR_HIDDEN_DIM},${CL_COND_ON_SUBJECT},,,,,,,${RUN_DIR}" >> "${ALL_RESULTS_FILE}"
        echo "  -> FAILED preflight (gamma_cl must be strictly > all other weights), continuing."
        continue
    fi

    set +e
    PYTHON_BIN="${PYTHON_BIN}" \
    OUTPUT_DIR="${OUTPUT_ROOT}" \
    RUN_ID="${RUN_ID}" \
    WANDB_COLLECTION="runs_20exp" \
    GAMMA_CL="${GAMMA_CL}" \
    Z_S_DIM="${Z_S_DIM}" \
    Z_I_DIM="${Z_I_DIM}" \
    BETA_S="${BETA_S}" \
    BETA_IS="${BETA_IS}" \
    BETA_I="${BETA_I}" \
    BETA_N="${BETA_N}" \
    LAMBDA_RECON="${LAMBDA_RECON}" \
    LAMBDA_SUBJ_CLS="${LAMBDA_SUBJ_CLS}" \
    LAMBDA_SUBJ_ADV="${LAMBDA_SUBJ_ADV}" \
    GRL_LAMBDA="${GRL_LAMBDA}" \
    C_MAX="${C_MAX}" \
    C_STOP_ITER="${C_STOP_ITER}" \
    IVAE_HIDDEN_DIM="${IVAE_HIDDEN_DIM}" \
    IVAE_N_LAYERS="${IVAE_N_LAYERS}" \
    IMAGE_PRIOR_HIDDEN_DIM="${IMAGE_PRIOR_HIDDEN_DIM}" \
    Z_IS_DIM="${Z_IS_DIM}" \
    Z_N_DIM="${Z_N_DIM}" \
    CL_COND_ON_SUBJECT="${CL_COND_ON_SUBJECT}" \
    bash "${BASE_SCRIPT}"
    exit_code=$?
    set -e

    if [ "${exit_code}" -ne 0 ]; then
        echo "${NOW},${EXP_ID},${RUN_ID},FAILED(${exit_code}),${GAMMA_CL},${Z_S_DIM},${Z_IS_DIM},${Z_I_DIM},${Z_N_DIM},${BETA_S},${BETA_IS},${BETA_I},${BETA_N},${LAMBDA_RECON},${LAMBDA_SUBJ_CLS},${LAMBDA_SUBJ_ADV},${GRL_LAMBDA},${C_MAX},${C_STOP_ITER},${IVAE_HIDDEN_DIM},${IVAE_N_LAYERS},${IMAGE_PRIOR_HIDDEN_DIM},${CL_COND_ON_SUBJECT},,,,,,,${RUN_DIR}" >> "${ALL_RESULTS_FILE}"
        echo "  -> FAILED (exit=${exit_code}), continuing to next experiment."
        continue
    fi

    if [ ! -f "${RUN_DIR}/results.csv" ]; then
        echo "${NOW},${EXP_ID},${RUN_ID},FAILED(no_results_csv),${GAMMA_CL},${Z_S_DIM},${Z_IS_DIM},${Z_I_DIM},${Z_N_DIM},${BETA_S},${BETA_IS},${BETA_I},${BETA_N},${LAMBDA_RECON},${LAMBDA_SUBJ_CLS},${LAMBDA_SUBJ_ADV},${GRL_LAMBDA},${C_MAX},${C_STOP_ITER},${IVAE_HIDDEN_DIM},${IVAE_N_LAYERS},${IMAGE_PRIOR_HIDDEN_DIM},${CL_COND_ON_SUBJECT},,,,,,,${RUN_DIR}" >> "${ALL_RESULTS_FILE}"
        echo "  -> FAILED (missing ${RUN_DIR}/results.csv), continuing."
        continue
    fi

    metrics="$(python3 - <<'PY' "${RUN_DIR}/results.csv"
import pandas as pd
import sys

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
for col in ["top1 acc", "top5 acc", "best top1 acc", "best top5 acc", "best test contrastive loss", "best test loss", "best epoch"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

means = [
    df["top1 acc"].mean(),
    df["top5 acc"].mean(),
    df["best top1 acc"].mean(),
    df["best top5 acc"].mean(),
    df["best test contrastive loss"].mean(),
    df["best test loss"].mean(),
    df["best epoch"].mean(),
]
print(",".join(f"{x:.4f}" for x in means))
PY
)"

    echo "${NOW},${EXP_ID},${RUN_ID},OK,${GAMMA_CL},${Z_S_DIM},${Z_IS_DIM},${Z_I_DIM},${Z_N_DIM},${BETA_S},${BETA_IS},${BETA_I},${BETA_N},${LAMBDA_RECON},${LAMBDA_SUBJ_CLS},${LAMBDA_SUBJ_ADV},${GRL_LAMBDA},${C_MAX},${C_STOP_ITER},${IVAE_HIDDEN_DIM},${IVAE_N_LAYERS},${IMAGE_PRIOR_HIDDEN_DIM},${CL_COND_ON_SUBJECT},${metrics},${RUN_DIR}" >> "${ALL_RESULTS_FILE}"
    echo "  -> Completed. Appended mean metrics to ${ALL_RESULTS_FILE}"
done

echo ""
echo "All scheduled experiments processed."
echo "See aggregated results: ${ALL_RESULTS_FILE}"
