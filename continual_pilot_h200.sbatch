#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --job-name=heart-pilot
#SBATCH --output=/project/yli4/ypan/heart_pilot_logs/%x-%j.out
#SBATCH --error=/project/yli4/ypan/heart_pilot_logs/%x-%j.err
# =============================================================================
# HEART H200 LIPID CONTINUAL pretraining: a completed HEART general checkpoint
# is continued on a weighted mixture of the Uni-Mol ligand LMDB and the
# enumerated Ugi-4CR lipid LMDBs (task unimol_contrastive_continual).
#
# Infrastructure is taken from the proven general launcher
# scripts/pretrain_heart-h200.sh, which this script does not modify or
# replace: same heart_lib.sh helpers (env activation, Uni-Core entry point,
# GPU/disk/CPU pre-flight, logging, fp16, early checkpoint, SIGUSR1/SIGTERM,
# cumulative --stop-time-hours, provenance, memory monitor) and the same CMRI
# allocation (1 H200, 16 CPUs, 200G, batch 448, update-freq 1, 4 workers).
# No LUMI A100/SLURM/conda setting is imported.
#
# Science is LUMI-lab model/pretrain/continual_pretrain.sh @ 611e02b:
#   lr 2e-5, weight decay 1e-4, polynomial decay, warmup 1000, 200000 updates,
#   masked token/coord/dist 0.5/1/5, x-norm 0.01, delta-pair-repr-norm 0.01,
#   contrastive 10.0, mask prob 0.15, uniform noise 1.0, seed 1, validation
#   and checkpoints every 1000 updates, data weights general 0.5 / lipid 1.0.
# A data weight is the fraction of that dataset used per epoch (LUMI
# semantics), not a mini-batch ratio; the real sampled counts are logged and
# written to <run dir>/mixture_report.json.
#
# TWO CHECKPOINT WORKFLOWS
#   New continual stage (no checkpoint_last.pt in the run dir):
#     * HEART_GENERAL_CHECKPOINT is opened READ-ONLY and never modified;
#     * tools/init_continual_checkpoint.py strictly checks every parameter
#       name/shape and the dictionary against a freshly built continual
#       model, then writes <run dir>/continual_init.pt with the model
#       weights only (no optimizer state, no LR-scheduler state, no fp16
#       scaler, no update counter, and no previous_training_time, which
#       Uni-Core would otherwise restore and which would make this job stop
#       immediately against its own --stop-time-hours);
#     * Uni-Core loads it through --finetune-from-model, so the optimizer,
#       the polynomial schedule over 200000 updates, the fp16 loss scaler and
#       the update counter all start from zero.
#   Resuming the continual stage (checkpoint_last.pt present):
#     * model + optimizer + LR scheduler + fp16 scaler + update counter +
#       data iterator state are restored from the continual run directory;
#     * --finetune-from-model is NOT passed, the stage marker and the
#       checkpoint's own task/schedule are verified, and tools/
#       check_resume_args.py refuses a changed scientific setting. The script
#       never silently restarts from scratch or re-initialises from the
#       general checkpoint when a resume fails.
#
# Output isolation: runs/h200/continual/<RUN_NAME>. The general H200 run
# directory runs/h200/full/* is refused as an output directory.
#
# Submit:
#   cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
#   mkdir -p ../../runs/h200/slurm
#   HEART_CONFIRM_CONTINUAL=1 \
#   HEART_GENERAL_CHECKPOINT=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/h200/full/heart-h200-b448-v1/checkpoint_best.pt \
#   HEART_LIPID_SPLITS=/home/ypan/heart_liver/model_train/lipid_library/lmdb/splits.json \
#   sbatch --export=ALL scripts/continual_pretrain_heart-h200.sh
# Resume after a time limit: the same command with the same RUN_NAME.
# Pre-flight only (no GPU training):
#   HEART_DRY_RUN=1 HEART_CONFIRM_CONTINUAL=1 HEART_GENERAL_CHECKPOINT=... \
#   HEART_LIPID_SPLITS=... bash scripts/continual_pretrain_heart-h200.sh
#
# Optional overrides: RUN_NAME, BATCH_SIZE (448), UPDATE_FREQ (1), NUM_WORKERS
#   (4), MIN_GPU_MEM_GB (120), MIN_FREE_GB (50), STOP_MARGIN_MINUTES (30),
#   EARLY_CHECKPOINT_UPDATES (1000; 0 = off), N_GPU (1; >1 untested),
#   GENERAL_DATA_WEIGHT (0.5), LIPID_DATA_WEIGHT (1.0), MAX_STEPS (200000),
#   HEART_ALLOW_ARG_CHANGE=1, HEART_MEM_INTERVAL, HEART_EXTRA_ARGS,
#   HEART_CONTINUAL_SMOKE_UPDATES=N (short isolated smoke run; RUN_NAME must contain "smoke"),
#   HEART_DATA_CHECK_SAMPLE (records sampled per LMDB by the data preflight, 64),
#   HEART_DATA_CHECK_SHARDS (lipid shards sampled per split, one per stratum of
#   ALL shard ids, 256), MAX_OPEN_LMDB_ENVS (64: per-process bound on open
#   LMDB environments of the mixed loader; v2 review P0).
# =============================================================================
# Permit direct Slurm submission only under the existing HEART v2 resource-profile guard.
export HEART_V2_ALLOW_DIRECT=1
set -eo pipefail

# >>> H200 paths: explicitly override old /scratch/ypan terminal settings >>>
# Identical to pretrain_heart-h200.sh on CMRI. The only difference: with the
# existing local-testing hook HEART_SKIP_ENV=1 (never used on CMRI) the caller's
# HEART_REPO/HEART_OUTPUT_ROOT/HEART_DATA_PATH are honoured, so the Windows
# test suite can run this launcher end-to-end on a tiny fixture.
if [ "${HEART_SKIP_ENV:-0}" = "1" ]; then
    export HEART_REPO="${HEART_REPO:-/home/ypan/heart_liver/model_train/heart-molecular-pretraining}"
    export HEART_OUTPUT_ROOT="${HEART_OUTPUT_ROOT:-$HEART_REPO/runs/h200}"
    export HEART_UNICORE_SRC="${HEART_UNICORE_SRC:-/home/ypan/heart_liver/model_train/third_party/Uni-Core}"
    export HEART_DATA_PATH="${HEART_DATA_PATH:-/home/ypan/heart_liver/model_train/UniMolReference/ligand_data/ligands}"
else
    # v2: the code that runs is THIS checkout (resolved from the submitted script
    # path, which sbatch records), never the historical clone
    # /home/ypan/heart_liver/model_train/heart-molecular-pretraining. bin/heart
    # exports HEART_REPO / HEART_OUTPUT_ROOT / HEART_DATA_PATH / HEART_UNICORE_SRC
    # from config/heart_v2.yaml; stale terminal values are refused below.
    v2_self="${BASH_SOURCE[0]}"
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        v2_sub="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p' | head -1)"
        v2_sub="${v2_sub%% *}"
        [ -n "$v2_sub" ] && [ -f "$v2_sub" ] && v2_self="$v2_sub"
    fi
    v2_repo="$(cd "$(dirname "$v2_self")/../../.." 2>/dev/null && pwd || true)"
    if [ ! -d "$v2_repo/model/unimol" ]; then
        v2_repo="${HEART_REPO:-}"
    fi
    [ -n "$v2_repo" ] && [ -d "$v2_repo/model/unimol" ] \
        || { echo "ERROR: cannot locate the HEART v2 checkout (submit with bin/heart)" >&2; exit 1; }
    if [ -n "${HEART_REPO:-}" ] && [ "$(cd "$HEART_REPO" 2>/dev/null && pwd)" != "$v2_repo" ]; then
        echo "ERROR: HEART_REPO=$HEART_REPO is not the checkout this script belongs to ($v2_repo)" >&2; exit 1
    fi
    export HEART_REPO="$v2_repo"
    export HEART_OUTPUT_ROOT="${HEART_OUTPUT_ROOT:-$HEART_REPO/runs/h200}"
    case "$HEART_OUTPUT_ROOT" in /scratch/*) echo "ERROR: stale HEART_OUTPUT_ROOT=$HEART_OUTPUT_ROOT" >&2; exit 1 ;; esac
    export HEART_UNICORE_SRC="${HEART_UNICORE_SRC:-/home/ypan/heart_liver/model_train/third_party/Uni-Core}"
    export HEART_DATA_PATH="${HEART_DATA_PATH:-/home/ypan/heart_liver/model_train/UniMolReference/ligand_data/ligands}"
fi
export BATCH_SIZE="${BATCH_SIZE:-448}"

# >>> Locate the repository (sbatch runs a spooled copy of this file) >>>
script_path="${BASH_SOURCE[0]}"
if [ -z "${HEART_REPO:-}" ] && [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p' | head -1)"
    submitted="${submitted%% *}"
    [ -n "$submitted" ] && [ -f "$submitted" ] && script_path="$submitted"
fi
lib="$(dirname "$script_path")/heart_lib.sh"
[ -f "$lib" ] || lib="${HEART_REPO:-/home/ypan/heart_liver/model_train/heart-molecular-pretraining}/model/pretrain/scripts/heart_lib.sh"
# shellcheck source=heart_lib.sh
. "$lib"
heart_find_repo "$script_path"
heart_paths

# >>> Explicit confirmation and required inputs (checked before anything else) >>>
[ "${HEART_CONFIRM_CONTINUAL:-0}" = "1" ] || die "continual pretraining requires HEART_CONFIRM_CONTINUAL=1"
[ -n "${BATCH_SIZE:-}" ] || die "BATCH_SIZE is empty; expected 448 for H200"
general_ckpt="${HEART_GENERAL_CHECKPOINT:-}"
pilot_variant="${HEART_PILOT_VARIANT:-}"
case "$pilot_variant" in A|B) ;; *) echo "ERROR: HEART_PILOT_VARIANT must be A or B" >&2; exit 2;; esac
pilot_lnpdb_splits="${HEART_LNPDB_SPLITS:-}"
if [ "$pilot_variant" = B ]; then
    [ -n "$pilot_lnpdb_splits" ] && [ -f "$pilot_lnpdb_splits" ] || { echo "ERROR: HEART_LNPDB_SPLITS required for B" >&2; exit 2; }
fi
pilot_fraction="${HEART_LNPDB_LIPID_FRACTION:-0.05}"
lipid_splits="${HEART_LIPID_SPLITS:-}"
lipid_lmdb="${HEART_LIPID_LMDB:-}"
[ -n "$general_ckpt" ] || die "HEART_GENERAL_CHECKPOINT=<completed general checkpoint> is required"
[ -f "$general_ckpt" ] || die "HEART_GENERAL_CHECKPOINT not found: $general_ckpt"
[ -r "$general_ckpt" ] || die "HEART_GENERAL_CHECKPOINT is not readable: $general_ckpt"
if [ -n "$lipid_splits" ]; then
    [ -f "$lipid_splits" ] || die "HEART_LIPID_SPLITS not found: $lipid_splits"
elif [ -n "$lipid_lmdb" ]; then
    warn "HEART_LIPID_LMDB is a smoke-test hook: it uses the same lipid LMDB(s) for train and valid"
else
    die "HEART_LIPID_SPLITS=<lipid lmdb dir>/splits.json is required (or HEART_LIPID_LMDB for a smoke test)"
fi

# >>> Heart Slurm information >>>
heart_slurm_info
heart_print_paths
log "general checkpoint (read-only): $general_ckpt"
log "lipid data : ${lipid_splits:-$lipid_lmdb}"

# >>> conda initialize (heart_gpu; this script never installs packages) >>>
heart_activate_env
heart_runtime_env
# v2: the allocation must be the continual profile of config/heart_v2.yaml
# (1 H200, 16 CPUs, 200G); never the 250-CPU library profile, never a T4.
(cd "$PRETRAIN_DIR" && "$PY" -m heart_v2.guard --profile continual) \
    || die "HEART v2 resource guard failed for the continual profile"
# v2: optional gate on the stage-05 receipt (bin/heart sets HEART_REQUIRE_LIBRARY_READY=1)
if [ "${HEART_REQUIRE_LIBRARY_READY:-0}" = "1" ] && [ -n "$lipid_splits" ]; then
    (cd "$PRETRAIN_DIR" && "$PY" -m lipid_library.ready verify "$(dirname "$lipid_splits")") \
        || die "the lipid library has no valid library_ready.json (stage 05 incomplete or artefacts changed)"
fi

# >>> HEART continual parameters (LUMI continual_pretrain.sh reference values) >>>
run_name="${RUN_NAME:-heart-pilot-${pilot_variant,,}}"
# Short, isolated H200 smoke run on real data (review P1 "production readiness"):
# HEART_CONTINUAL_SMOKE_UPDATES=N trains N updates of the real configuration into
# a run directory whose name must contain "smoke"; production runs never reuse it.
smoke_updates="${HEART_CONTINUAL_SMOKE_UPDATES:-}"
if [ -n "$smoke_updates" ]; then
    case "$smoke_updates" in ''|*[!0-9]*|0) die "HEART_CONTINUAL_SMOKE_UPDATES must be a positive integer" ;; esac
    run_name="${RUN_NAME:-heart-pilot-${pilot_variant,,}-smoke}"
    case "$run_name" in *smoke*) ;; *) die "a smoke run needs a RUN_NAME containing 'smoke' (got $run_name)" ;; esac
else
    case "$run_name" in *smoke*) die "RUN_NAME $run_name is reserved for HEART_CONTINUAL_SMOKE_UPDATES runs" ;; esac
fi
n_gpu="${N_GPU:-1}"
num_workers="${NUM_WORKERS:-4}"        # H200 default (proven by the general H200 runs)
max_open_lmdb_envs="${MAX_OPEN_LMDB_ENVS:-64}"   # v2 review P0: LRU bound per process
lr=2e-5                                # LUMI continual
wd=1e-4
batch_size="$BATCH_SIZE"
batch_size_valid="$batch_size"
update_freq="${UPDATE_FREQ:-1}"
heads=64
masked_token_loss=0.5                  # LUMI continual
masked_coord_loss=1                    # LUMI continual
masked_dist_loss=5                     # LUMI continual
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
contrastive_loss=10.0
contrastive_view_mode=independent      # HEART P3 (LUMI's mixed task gives identical views)
invalid_smiles_policy=skip-2d          # HEART P6
max_invalid_smiles_fraction=0.01       # HEART P6
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
max_atoms=256
seed=1
warmup_steps=1000                      # LUMI continual
max_steps="${MAX_STEPS:-200000}"       # LUMI continual
general_data_weight="${GENERAL_DATA_WEIGHT:-0.5}"   # LUMI continual
lipid_data_weight="${LIPID_DATA_WEIGHT:-1.0}"       # LUMI continual
log_interval=100
validate_interval_updates=1000         # LUMI continual
save_interval_updates=1000             # LUMI continual
keep_interval_updates=10               # + checkpoint_last.pt + checkpoint_best.pt
early_checkpoint_updates="${EARLY_CHECKPOINT_UPDATES:-1000}"
[ -n "$smoke_updates" ] && early_checkpoint_updates="${EARLY_CHECKPOINT_UPDATES:-0}"
stop_margin_minutes="${STOP_MARGIN_MINUTES:-30}"
min_gpu_mem_gb="${MIN_GPU_MEM_GB:-120}"
min_free_gb="${MIN_FREE_GB:-50}"

for v in batch_size update_freq n_gpu num_workers stop_margin_minutes min_free_gb early_checkpoint_updates max_steps; do
    case "${!v}" in ''|*[!0-9]*) die "$v must be a non-negative integer (got '${!v}')" ;; esac
done
case "$min_gpu_mem_gb" in ''|*[!0-9.]*|*.*.*) die "MIN_GPU_MEM_GB must be a non-negative number (got '$min_gpu_mem_gb')" ;; esac
for v in general_data_weight lipid_data_weight; do
    case "${!v}" in ''|*[!0-9.]*|*.*.*) die "$v must be a number in (0, 1] (got '${!v}')" ;; esac
done
log "DataLoader workers: $num_workers$([ "$num_workers" = 0 ] && echo ' (data loaded in the main process)') | GPU memory required: >= $min_gpu_mem_gb GiB"
[ "$batch_size" -ge 1 ] && [ "$update_freq" -ge 1 ] && [ "$n_gpu" -ge 1 ] || die "BATCH_SIZE, UPDATE_FREQ and N_GPU must be >= 1"
effective_batch=$(( batch_size * update_freq * n_gpu ))
log "micro-batch $batch_size x update-freq $update_freq x GPUs $n_gpu = effective $effective_batch molecules/update"
log "NT-Xent negatives come from one micro-batch per GPU (gathered across GPUs): $(( 2 * batch_size * n_gpu - 2 )) per positive pair"
log "data weights: general $general_data_weight, lipid $lipid_data_weight (fraction of each dataset per epoch, NOT a mini-batch ratio)"
[ "$update_freq" -gt 1 ] && warn "update-freq $update_freq: accumulation reproduces the effective batch for the reconstruction losses, not for the contrastive negatives"

precision_args=(--fp16 --fp16-init-scale 4 --fp16-scale-window 256)
USE_CPU=0
if [ "${HEART_LOCAL_CPU_TEST:-0}" = "1" ]; then
    precision_args=(--cpu)             # local CPU test suite only; never on CMRI
    USE_CPU=1
fi
read -r -a extra_args <<< "${HEART_EXTRA_ARGS:-}"
heart_reject_forbidden_args "${extra_args[@]}"

DRY_RUN="${HEART_DRY_RUN:-0}"
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$DRY_RUN" != "1" ] && [ "$USE_CPU" != "1" ]; then
    die "continual pretraining runs only under sbatch (use HEART_DRY_RUN=1 for a login-node pre-flight)"
fi

# >>> Pre-flight: general dataset (read-only) and lipid LMDBs >>>
heart_check_data
# Fail closed on an incomplete/smoke lipid dataset or a conformer-slot policy that
# differs between general and lipid records (review P0-1, P1 "2D policy").
smoke_data_args=()
if [ -n "$lipid_lmdb" ] || [ "${HEART_ALLOW_SMOKE_LIPID:-0}" = "1" ]; then
    [ "$USE_CPU" = "1" ] || [ "$DRY_RUN" = "1" ] || die "smoke lipid data (HEART_LIPID_LMDB / HEART_ALLOW_SMOKE_LIPID) is for local tests and dry runs only"
    smoke_data_args=(--allow-smoke)
fi
# Pilot-only threshold: accept validated records with partial 2D fallback slots.
# This does not change the underlying LMDB or the model's sampling weights.
# Runtime checks still verify metadata, record integrity and dominant profile compatibility.
data_check_args=(--general "$DATA_PATH" --sample "${HEART_DATA_CHECK_SAMPLE:-64}"
    --lipid-shards "${HEART_DATA_CHECK_SHARDS:-256}"
    --min-dominant-fraction "${HEART_PILOT_MIN_DOMINANT:-0.85}")
if [ -n "$lipid_splits" ]; then data_check_args+=(--lipid-splits "$lipid_splits"); else data_check_args+=(--lipid-lmdb "$lipid_lmdb"); fi
"$PY" "$PRETRAIN_DIR/tools/check_continual_data.py" "${data_check_args[@]}" "${smoke_data_args[@]}"     || die "continual data preflight failed (incomplete/smoke lipid data or mismatched conformer slot policy)"

# >>> Fixed continual run directory: resubmission resumes; never the general run dir >>>
save_dir="$OUTPUT_ROOT/continual/$run_name"
case "$(realpath -m "$save_dir")/" in
    */runs/h200/full/*) die "refusing to write inside the general H200 run tree: $save_dir" ;;
esac
if [ "$DRY_RUN" = "1" ]; then
    # v2: a dry run (bin/heart continual --check) is strictly read-only. The frozen
    # heart_check_save_dir creates OUTPUT_ROOT and touch-tests a probe file, so the same
    # policy (not inside/containing the dataset; writable) is checked without writing.
    v2_save_abs="$(realpath -m "$save_dir")"; v2_data_abs="$(realpath -m "$DATA_PATH")"
    case "$v2_save_abs/" in "$v2_data_abs"/*) die "save dir must not be inside the dataset directory" ;; esac
    case "$v2_data_abs/" in "$v2_save_abs"/*) die "save dir must not contain the dataset" ;; esac
    v2_probe="$OUTPUT_ROOT/continual"
    while [ ! -d "$v2_probe" ]; do v2_probe="$(dirname "$v2_probe")"; done
    [ -w "$v2_probe" ] || die "$v2_probe (nearest existing parent of the run dir) is not writable"
    free_gb="$(df -Pk "$v2_probe" | awk 'NR==2 {print int($4/1048576)}')"
    log "dry run: nothing created; nearest existing parent $v2_probe is writable"
else
    heart_check_save_dir "$save_dir"
    mkdir -p "$OUTPUT_ROOT/continual"
    free_gb="$(df -Pk "$OUTPUT_ROOT/continual" | awk 'NR==2 {print int($4/1048576)}')"
fi
log "free disk under $OUTPUT_ROOT/continual: ${free_gb} GiB (need >= $min_free_gb)"
[ "$free_gb" -ge "$min_free_gb" ] || die "less than $min_free_gb GiB free for checkpoints (~0.6-0.8 GB each, up to 12 kept)"
# >>> File-descriptor budget (v2 review P0) >>>
# Each process (trainer + NUM_WORKERS loader workers, per GPU) holds at most
# MAX_OPEN_LMDB_ENVS LMDB environments of the mixed dataset (1 fd each with
# lock=False); 512 fds per process are reserved for torch/NCCL/logging.
if [ "$max_open_lmdb_envs" = "auto" ]; then
    # v2: keep every lipid shard open when the hard fd limit allows it (an LRU miss
    # re-opens an LMDB per sample with ~3,000 shards); otherwise the largest bound that fits.
    n_lipid_lmdb="$("$PY" -c 'import json, sys
s = json.load(open(sys.argv[1])) if sys.argv[1] else {"train": [], "valid": []}
print(max(len(s.get("train", [])), len(s.get("valid", []))) + 2)' "$lipid_splits" 2>/dev/null || echo 64)"
    fd_hard="$(ulimit -Hn)"
    [ "$fd_hard" = "unlimited" ] && fd_hard=1048576
    fit=$(( (fd_hard - 512) / 2 ))
    max_open_lmdb_envs=$(( n_lipid_lmdb < fit ? n_lipid_lmdb : fit ))
    [ "$max_open_lmdb_envs" -ge 1 ] || max_open_lmdb_envs=64
    log "MAX_OPEN_LMDB_ENVS=auto -> $max_open_lmdb_envs (lipid LMDBs per split + 2: $n_lipid_lmdb, hard fd limit $fd_hard)"
fi
case "$max_open_lmdb_envs" in ''|*[!0-9]*|0) die "MAX_OPEN_LMDB_ENVS must be a positive integer" ;; esac
fd_need=$(( 2 * max_open_lmdb_envs + 512 ))   # train + valid mixed datasets per process
fd_soft="$(ulimit -Sn)"
if [ "$fd_soft" != "unlimited" ] && [ "$fd_soft" -lt "$fd_need" ]; then
    ulimit -Sn "$fd_need" 2>/dev/null || ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true
    fd_soft="$(ulimit -Sn)"
fi
log "open-file limit $fd_soft per process; mixed loader keeps <= $max_open_lmdb_envs LMDB envs open per process (need $fd_need)"
if [ "$fd_soft" != "unlimited" ] && [ "$fd_soft" -lt "$fd_need" ]; then
    die "ulimit -n $fd_soft < $fd_need: lower MAX_OPEN_LMDB_ENVS or raise the open-file limit"
fi
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "$num_workers" -ge "$SLURM_CPUS_PER_TASK" ]; then
    die "NUM_WORKERS=$num_workers must be < SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
fi
case "$(realpath -m "$general_ckpt")" in
    "$(realpath -m "$save_dir")"/*) die "the general checkpoint must not live inside the continual run dir" ;;
esac

# >>> Which of the two checkpoint workflows is this? >>>
resume_from=""
stage_file="$save_dir/continual_stage.json"
init_ckpt="$save_dir/continual_init.pt"
if [ -f "$save_dir/checkpoint_last.pt" ]; then
    resume_from="$(heart_ckpt_updates "$save_dir/checkpoint_last.pt")" || die "cannot read $save_dir/checkpoint_last.pt"
    [ "$resume_from" -lt "$max_steps" ] || die "continual run already reached $resume_from updates (max $max_steps)"
    log "run dir    : $save_dir (RESUMING the continual stage from checkpoint_last.pt @ $resume_from updates)"
    [ -f "$stage_file" ] || warn "no $stage_file: this run directory was not created by this script"
else
    log "run dir    : $save_dir (NEW continual stage from $general_ckpt)"
fi

# >>> Pre-flight: GPU / PyTorch / Uni-Core registry / time >>>
gpu_optional=0
[ "$DRY_RUN" = "1" ] && [ -z "${SLURM_JOB_ID:-}" ] && gpu_optional=1
heart_gpu_preflight "$USE_CPU" "$min_gpu_mem_gb" "$gpu_optional"
heart_registry_check
(cd "$MODEL_ROOT" && "$PY" pretrain/tools/check_continual_registry.py --user-dir ./unimol) \
    || die "Uni-Core cannot discover the unimol_contrastive_continual task"
heart_train_entry
(cd "$MODEL_ROOT" && "$PY" -c 'import unicore; from unicore import utils; from types import SimpleNamespace; from unicore.tasks import TASK_REGISTRY; utils.import_user_module(SimpleNamespace(user_dir="./unimol")); assert "unimol_contrastive_pilot" in TASK_REGISTRY, "pilot task not registered"; print("pilot task registered")') \
    || die "Pilot task not discoverable"
[ "$DRY_RUN" = "1" ] || mkdir -p "$save_dir"
heart_stop_time_args "$save_dir" "$stop_margin_minutes"
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${#STOP_ARGS[@]}" -eq 0 ] && [ "$USE_CPU" != "1" ]; then
    die "could not derive --stop-time-hours from the SLURM time limit; refusing an unguarded continual run"
fi

# >>> Lipid data arguments >>>
lipid_args=()
if [ -n "$lipid_splits" ]; then
    lipid_args=(--lipid-splits "$lipid_splits")
    [ "${#smoke_data_args[@]}" -gt 0 ] && lipid_args+=(--allow-smoke-lipid-data)
else
    lipid_args=(--lipid-lmdb "$lipid_lmdb")
fi

# >>> Lipid data identity pinned to the run (v2 review: safe resume) >>>
# The run directory records the sha256 of splits.json (which itself pins the
# manifest hash, i.e. every shard's LMDB sha256 and the train/valid assignment).
# A resubmission whose splits.json differs -- e.g. stage 05 re-run with another
# SPLIT_SEED -- would silently change the training mixture and the validation
# set mid-run, so it is refused (HEART_ALLOW_ARG_CHANGE=1 overrides, documented).
lipid_identity="$("$PY" -c 'import hashlib, sys
p = sys.argv[1]
print(hashlib.sha256(open(p, "rb").read()).hexdigest() if p else "lipid-lmdb:" + sys.argv[2])' \
    "$lipid_splits" "$lipid_lmdb")" || die "cannot hash the lipid splits"
identity_file="$save_dir/lipid_data_identity.txt"
if [ -f "$identity_file" ]; then
    if [ "$(cat "$identity_file")" != "$lipid_identity" ]; then
        if [ "${HEART_ALLOW_ARG_CHANGE:-0}" = "1" ]; then
            warn "lipid data (splits.json) differs from the one this run started with (HEART_ALLOW_ARG_CHANGE=1)"
        else
            die "the lipid splits.json differs from the one this run started with ($identity_file): a resume must use the same train/valid shards"
        fi
    fi
elif [ -n "$resume_from" ]; then
    warn "no $identity_file in a resumed run directory: recording the current lipid data identity"
fi
if [ "$DRY_RUN" != "1" ]; then
    printf "%s\n" "$lipid_identity" > "$identity_file.tmp" && mv -f "$identity_file.tmp" "$identity_file"
fi

# >>> Pilot dataset identity; prevent accidental resume against changed LNPDB >>>
if [ "$pilot_variant" = B ]; then
    pilot_lnpdb_hash="$(sha256sum "$pilot_lnpdb_splits" | awk '{print $1}')"
    pilot_lnpdb_identity="$save_dir/lnpdb_data_identity.txt"
    if [ -f "$pilot_lnpdb_identity" ] && [ "$(cat "$pilot_lnpdb_identity")" != "$pilot_lnpdb_hash" ]; then
        die "LNPDB splits changed since pilot began; refusing to resume"
    fi
    if [ "$DRY_RUN" != "1" ]; then
        printf '%s\n' "$pilot_lnpdb_hash" > "$pilot_lnpdb_identity"
    fi
fi

# >>> Training command >>>
cd "$MODEL_ROOT"
# Independent registered pilot task; leaves original continual task untouched.
pilot_task=unimol_contrastive_pilot
pilot_extra=(--pilot-variant "$pilot_variant" --lnpdb-lipid-fraction "$pilot_fraction")
if [ -n "$pilot_lnpdb_splits" ]; then pilot_extra+=(--lnpdb-splits "$pilot_lnpdb_splits"); fi
CMD=("${TRAIN_ENTRY[@]}" "$DATA_PATH" --user-dir ./unimol --train-subset train --valid-subset valid
    --num-workers "$num_workers" --data-buffer-size 20 --ddp-backend=c10d
    --task "$pilot_task" --loss unimol_contrastive --arch unimol_contrastive_base
    --encoder-attention-heads "$heads"
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay "$wd"
    --lr-scheduler polynomial_decay --lr "$lr" --warmup-updates "$warmup_steps" --total-num-update "$max_steps"
    --update-freq "$update_freq" --seed "$seed"
    "${precision_args[@]}" --tensorboard-logdir "$save_dir/tsb"
    --max-update "$max_steps" --log-interval "$log_interval" --log-format simple
    --save-interval-updates "$save_interval_updates" --validate-interval-updates "$validate_interval_updates"
    --keep-interval-updates "$keep_interval_updates" --no-epoch-checkpoints
    --masked-token-loss "$masked_token_loss" --masked-coord-loss "$masked_coord_loss" --masked-dist-loss "$masked_dist_loss"
    --x-norm-loss "$x_norm_loss" --delta-pair-repr-norm-loss "$delta_pair_repr_norm_loss"
    --mask-prob "$mask_prob" --noise-type "$noise_type" --noise "$noise" --max-atoms "$max_atoms"
    --batch-size "$batch_size" --batch-size-valid "$batch_size_valid"
    --save-dir "$save_dir" --tmp-save-dir "$save_dir" --only-polar "$only_polar" --mode train
    --dict-name dict.txt --contrastive-loss "$contrastive_loss" --contrastive-view-mode "$contrastive_view_mode"
    --invalid-smiles-policy "$invalid_smiles_policy" --max-invalid-smiles-fraction "$max_invalid_smiles_fraction"
    "${lipid_args[@]}" "${pilot_extra[@]}" --general-data-weight "$general_data_weight" --lipid-data-weight "$lipid_data_weight"
    --mixture-report "$save_dir/mixture_report.json" --require-equal-coordinate-slots
    --max-open-lmdb-envs "$max_open_lmdb_envs"
    "${STOP_ARGS[@]}" "${extra_args[@]}")
if [ -n "$smoke_updates" ]; then
    CMD+=(--max-update "$smoke_updates" --save-interval-updates "$smoke_updates"
          --validate-interval-updates "$smoke_updates")
    log "SMOKE RUN: $smoke_updates updates of the production configuration into $save_dir"
fi

# >>> New stage: weights-only init checkpoint, strictly checked against THIS command >>>
if [ -z "$resume_from" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        log "HEART_DRY_RUN=1: checking $general_ckpt against the command (nothing written)"
        "$PY" pretrain/tools/init_continual_checkpoint.py "$general_ckpt" \
            --data "$DATA_PATH" --out "$init_ckpt" --user-dir ./unimol \
            --task "$pilot_task" --dict-name dict.txt --dry-run -- "${CMD[@]}" \
            || die "the general checkpoint is not compatible with the continual model"
    else
        log "checking $general_ckpt against the continual model this command builds (strict)"
        "$PY" pretrain/tools/init_continual_checkpoint.py "$general_ckpt" \
            --data "$DATA_PATH" --out "$init_ckpt" --user-dir ./unimol \
            --task "$pilot_task" --dict-name dict.txt \
            --report "$save_dir/continual_init_report.json" -- "${CMD[@]}" \
            || die "the general checkpoint is not compatible with the continual model (see $save_dir/continual_init_report.json)"
        [ -f "$init_ckpt" ] || die "init checkpoint was not written: $init_ckpt"
    fi
    CMD+=(--finetune-from-model "$init_ckpt")
else
    # Resuming: never re-initialise from the general checkpoint.
    log "resume: not passing --finetune-from-model; continual state comes from checkpoint_last.pt"
fi
if [ "$n_gpu" -gt 1 ]; then
    CMD=(torchrun --standalone --nproc_per_node="$n_gpu" "${CMD[@]}")
fi

echo "=== Command ==="
printf '%q ' "${CMD[@]}"; echo
if [ -n "$resume_from" ]; then
    allow=(); [ "${HEART_ALLOW_ARG_CHANGE:-0}" = "1" ] && allow=(--allow-change)
    "$PY" "$PRETRAIN_DIR/tools/check_resume_args.py" "$save_dir/checkpoint_last.pt" "${allow[@]}" -- "${CMD[@]}" \
        || die "resume settings differ from the continual checkpoint (HEART_ALLOW_ARG_CHANGE=1 to override, and document why)"
    "$PY" "$PRETRAIN_DIR/tools/check_continual_stage.py" "$save_dir/checkpoint_last.pt" \
        --expect-task unimol_contrastive_pilot --expect-total-num-update "$max_steps" \
        ${stage_file:+--stage-file "$stage_file"} \
        || die "checkpoint_last.pt in $save_dir is not a continual-stage checkpoint for this configuration"
fi

# Early checkpoint: segment 1 stops at early_checkpoint_updates, segment 2 resumes it.
target_update="$max_steps"
for i in "${!CMD[@]}"; do
    [ "${CMD[$i]}" = "--max-update" ] && target_update="${CMD[$((i + 1))]}"
done
early_segment=0
if [ "$early_checkpoint_updates" -gt 0 ] && [ "${resume_from:-0}" -lt "$early_checkpoint_updates" ] \
        && [ "$early_checkpoint_updates" -lt "$target_update" ]; then
    early_segment=1
    log "early checkpoint: segment 1 to $early_checkpoint_updates updates (validate + checkpoint_last.pt), then resume to $target_update in this job"
else
    log "early checkpoint: not needed (start ${resume_from:-0}, early $early_checkpoint_updates, target $target_update)"
fi
if [ "$DRY_RUN" = "1" ]; then
    log "HEART_DRY_RUN=1: pre-flight passed; would write to $save_dir; not training"
    exit 0
fi

# >>> Stage marker: records which general checkpoint this stage started from >>>
if [ ! -f "$stage_file" ]; then
    "$PY" - "$stage_file" "$general_ckpt" "$init_ckpt" "$max_steps" "$lr" "$warmup_steps" \
        "$general_data_weight" "$lipid_data_weight" "${lipid_splits:-$lipid_lmdb}" <<'PYEOF'
import json, os, sys
path, general, init, steps, lr, warmup, gw, lw, lipid = sys.argv[1:10]
json.dump({
    "stage": "continual",
    "task": "unimol_contrastive_pilot",
    "general_checkpoint": os.path.abspath(general),
    "continual_init_checkpoint": os.path.abspath(init),
    "total_num_update": int(steps),
    "lr": float(lr),
    "warmup_updates": int(warmup),
    "general_data_weight": float(gw),
    "lipid_data_weight": float(lw),
    "lipid_data": lipid,
}, open(path, "w"), indent=2, sort_keys=True)
PYEOF
    log "stage file : $stage_file"
fi

heart_provenance "$save_dir" "$script_path" "${CMD[@]}"
cp "$lib" "$save_dir/provenance/heart_lib.sh.used" 2>/dev/null || true
train_log="$save_dir/train-${SLURM_JOB_ID:-local}.log"
log "training log: $train_log"

# >>> Run in the background so SLURM signals reach this shell >>>
train_pid=""
got_term=0
on_signal() {
    log "received SIG$1: Uni-Core stops itself at --stop-time-hours (validate + checkpoint_last.pt)"
    if [ "$1" = TERM ]; then
        got_term=1
        if [ -n "$train_pid" ] && kill -0 "$train_pid" 2>/dev/null; then
            log "SIGTERM: forwarding to training (pid $train_pid); resume from the last saved checkpoint"
            kill -TERM "$train_pid" 2>/dev/null || true
        fi
    fi
}
# $@ = training command; sets rc
run_training() {
    "$@" > >(tee -a "$train_log") 2>&1 &
    train_pid=$!
    rc=0
    while :; do
        wait "$train_pid"
        rc=$?
        kill -0 "$train_pid" 2>/dev/null || break
    done
    train_pid=""
}
heart_memory_limits
mem_log="$save_dir/memory-${SLURM_JOB_ID:-local}.tsv"
MEM_MON_PID=""
[ "${HEART_MEM_INTERVAL:-60}" != 0 ] && heart_mem_monitor_start "$mem_log" "${HEART_MEM_INTERVAL:-60}"
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
set +e
run_main=1
if [ "$early_segment" = 1 ]; then
    log "segment 1/2: training to the early checkpoint at $early_checkpoint_updates updates"
    run_training "${CMD[@]}" --max-update "$early_checkpoint_updates"
    early_now="$(heart_ckpt_updates "$save_dir/checkpoint_last.pt" 2>/dev/null)"
    if [ "$rc" -ne 0 ] || [ "$got_term" = 1 ] || [ "$early_now" != "$early_checkpoint_updates" ]; then
        log "segment 1 ended (exit $rc, checkpoint_last.pt @ ${early_now:-none}); not continuing in this job"
        run_main=0
    else
        log "early checkpoint written: checkpoint_last.pt @ $early_now updates"
        # Segment 2 resumes the continual checkpoint: drop --finetune-from-model.
        CMD2=()
        skip_next=0
        for a in "${CMD[@]}"; do
            if [ "$skip_next" = 1 ]; then skip_next=0; continue; fi
            if [ "$a" = "--finetune-from-model" ]; then skip_next=1; continue; fi
            CMD2+=("$a")
        done
        if "$PY" "$PRETRAIN_DIR/tools/check_resume_args.py" "$save_dir/checkpoint_last.pt" -- "${CMD2[@]}"; then
            heart_stop_time_args "$save_dir" "$stop_margin_minutes"    # minus the time segment 1 used
            log "segment 2/2: resuming checkpoint_last.pt @ $early_now -> $target_update updates"
            CMD=("${CMD2[@]}")
        else
            log "resume settings differ from the early checkpoint; not continuing"
            rc=1
            run_main=0
        fi
    fi
fi
[ "$run_main" = 1 ] && run_training "${CMD[@]}" "${STOP_ARGS[@]}"
trap - USR1 TERM
heart_mem_monitor_stop "$mem_log"

echo "=== Summary ==="
"$PY" "$PRETRAIN_DIR/tools/summarize_log.py" "$train_log" || log "summary reported FAIL items (see above)"
ls -lh "$save_dir"
[ -f "$save_dir/mixture_report.json" ] && cat "$save_dir/mixture_report.json"
if [ -f "$save_dir/checkpoint_last.pt" ]; then
    now="$(heart_ckpt_updates "$save_dir/checkpoint_last.pt" || echo '?')"
    log "checkpoint_last.pt @ $now / $max_steps continual updates"
    [ "$now" != "$max_steps" ] && log "not finished: resubmit the same command to resume from $now updates"
fi
log "training exit code: $rc"
exit "$rc"
