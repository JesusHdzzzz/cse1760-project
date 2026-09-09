#!/usr/bin/env bash

# Run every current executable workflow in Parts 1–3.
#
# Behavior:
# - Runs sequentially so expensive jobs do not compete for RAM/CPU.
# - Creates a unique timestamped portfolio run.
# - Saves experiment artifacts beneath partN/outputs/.
# - Saves stdout/stderr for every command beneath logs/.
# - Continues to later experiments if one experiment fails.
# - Writes a final TSV summary with exit status and runtime.
#
# Run from anywhere:
#   bash experiment_runner.sh
#
# Optional:
#   PYTHON_BIN=/path/to/python bash experiment_runner.sh

set -uo pipefail

###############################################################################
# Repository / environment setup
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
PORTFOLIO_LABEL="portfolio-${RUN_ID}"

LOG_ROOT="${SCRIPT_DIR}/logs/${PORTFOLIO_LABEL}"
SUMMARY_FILE="${LOG_ROOT}/run_summary.tsv"

mkdir -p "$LOG_ROOT"

###############################################################################
# Helpers
###############################################################################

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

run_command() {
    local name="$1"
    shift

    local log_file="${LOG_ROOT}/${name}.log"
    local start_epoch
    local end_epoch
    local elapsed
    local status

    start_epoch="$(date +%s)"

    echo
    echo "======================================================================"
    echo "[$(timestamp)] START: ${name}"
    echo "Command:"
    printf '  %q' "$@"
    echo
    echo "Log: ${log_file}"
    echo "======================================================================"

    # python -u / normal executable output is streamed live and also saved.
    "$@" 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}

    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - start_epoch))

    if [[ $status -eq 0 ]]; then
        echo "[$(timestamp)] PASS: ${name} (${elapsed}s)"
        result="PASS"
    else
        echo "[$(timestamp)] FAIL: ${name} (${elapsed}s, exit=${status})"
        result="FAIL"
    fi

    printf "%s\t%s\t%s\t%s\t%s\n" \
        "$name" "$result" "$status" "$elapsed" "$log_file" \
        >> "$SUMMARY_FILE"

    # Deliberately return success so one failed experiment does not stop the full run.
    return 0
}

###############################################################################
# Run metadata
###############################################################################

printf "experiment\tresult\texit_code\truntime_seconds\tlog\n" > "$SUMMARY_FILE"

{
    echo "CSE 176 portfolio rerun"
    echo "Run ID:       ${RUN_ID}"
    echo "Started:      $(timestamp)"
    echo "Repository:   ${SCRIPT_DIR}"
    echo "Python:       ${PYTHON_BIN}"
    echo
    echo "Git:"
    git rev-parse HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
    echo
    echo "Python version:"
    "$PYTHON_BIN" --version 2>&1 || true
    echo
    echo "Java version:"
    java -version 2>&1 || true
    echo
    echo "Installed key packages:"
    "$PYTHON_BIN" - <<'PY' 2>&1
packages = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "xgboost",
    "h2o",
]
for package in packages:
    try:
        module = __import__(package)
        print(f"{package}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{package}: ERROR: {exc}")
PY
} | tee "${LOG_ROOT}/environment.log"

###############################################################################
# Output directories
###############################################################################

P1_ROOT="${SCRIPT_DIR}/part1/outputs/${PORTFOLIO_LABEL}"
P2_ROOT="${SCRIPT_DIR}/part2/outputs/${PORTFOLIO_LABEL}"
P3_ROOT="${SCRIPT_DIR}/part3/outputs/${PORTFOLIO_LABEL}"

mkdir -p "$P1_ROOT" "$P2_ROOT" "$P3_ROOT"

###############################################################################
# PART 1
###############################################################################

echo
echo "######################################################################"
echo "# PART 1"
echo "######################################################################"

# Utility / dataset inspection: stdout only.
run_command \
    "part1_feature" \
    "$PYTHON_BIN" -u part1/src/feature.py

run_command \
    "part1_logistic_regression" \
    "$PYTHON_BIN" -u part1/src/logistic_regression.py \
    --output-dir "${P1_ROOT}/logistic_regression"

run_command \
    "part1_random_forest" \
    "$PYTHON_BIN" -u part1/src/random_forest.py \
    --output-dir "${P1_ROOT}/random_forest"

###############################################################################
# PART 2
###############################################################################

echo
echo "######################################################################"
echo "# PART 2"
echo "######################################################################"

run_command \
    "part2_xgb_pixels" \
    "$PYTHON_BIN" -u part2/src/xgb_mnist_pixels.py \
    --output-dir "${P2_ROOT}/pixels"

run_command \
    "part2_xgb_lenet" \
    "$PYTHON_BIN" -u part2/src/xgb_mnist_lenet.py \
    --output-dir "${P2_ROOT}/lenet"

###############################################################################
# PART 3 — non-H2O / cheaper work first
###############################################################################

echo
echo "######################################################################"
echo "# PART 3 — RANDOM FOREST / PLOTS"
echo "######################################################################"

run_command \
    "part3_stroke_random_forest" \
    "$PYTHON_BIN" -u part3/src/stroke_random_forest.py \
    --output-dir "${P3_ROOT}/stroke_random_forest"

run_command \
    "part3_target_distribution" \
    "$PYTHON_BIN" -u part3/src/histogram_boxplot.py \
    --output-dir "${P3_ROOT}/target_distribution"

###############################################################################
# PART 3 — expensive H2O workflows
###############################################################################

echo
echo "######################################################################"
echo "# PART 3 — H2O"
echo "######################################################################"

run_command \
    "part3_stroke_h2o_gbm" \
    "$PYTHON_BIN" -u part3/src/stroke_h2o_gbm.py \
    --max-models 30 \
    --output-dir "${P3_ROOT}/stroke_h2o_gbm"

run_command \
    "part3_supercon_gbm" \
    "$PYTHON_BIN" -u part3/src/supercon_gbm.py \
    --run-gbm-sweep \
    --output-dir "${P3_ROOT}/supercon_gbm"

###############################################################################
# PART 3 — optional/non-deployable diagnostic
###############################################################################

run_command \
    "part3_supercon_target_conditioned_diagnostic" \
    "$PYTHON_BIN" -u part3/src/supercon_split_by_tc_gbm.py \
    --output-dir "${P3_ROOT}/supercon_target_conditioned_diagnostic"

###############################################################################
# Final summary
###############################################################################

FINISHED_AT="$(timestamp)"

echo
echo "======================================================================"
echo "ALL COMMANDS FINISHED"
echo "Finished: ${FINISHED_AT}"
echo
echo "Run outputs:"
echo "  Part 1: ${P1_ROOT}"
echo "  Part 2: ${P2_ROOT}"
echo "  Part 3: ${P3_ROOT}"
echo
echo "Logs:"
echo "  ${LOG_ROOT}"
echo
echo "Summary:"
echo "======================================================================"

if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY_FILE"
else
    cat "$SUMMARY_FILE"
fi

echo
echo "Failed experiments:"
awk -F '\t' 'NR > 1 && $2 == "FAIL" { print "  - " $1 " (exit " $3 ")" }' \
    "$SUMMARY_FILE"

echo
echo "Done."
