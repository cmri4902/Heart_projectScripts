#!/usr/bin/env bash
# Finalize and verify the existing LNPDB LMDB shards; preserve timestamped logs.
# This script NEVER regenerates conformers or changes the original Ugi LMDB.
set -Eeuo pipefail

REPO="${REPO:-/home/ypan/heart_liver/model_train/heart-molecular-pretraining}"
ROOT="${LNPDB_ROOT:-/project/yli4/ypan/lipid_library/lnpdb_heart}"
OUT="${LNPDB_FULL_OUT:-$ROOT/lmdb_full}"
PREPARED="${LNPDB_PREPARED:-$ROOT/prepared/lnpdb_unique_prepared.csv}"
DICT="${GENERAL_DICT:-/project/yli4/ypan/UniMolReference/ligand_data/ligands/dict.txt}"
PY="${PY:-$HOME/.conda/envs/heart_gpu/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-16}"

# Detect the actual adapter location. Keep existing HPC layout unchanged.
if [[ -z "${ADAPTER:-}" ]]; then
  for candidate in \
    "$REPO/model/pretrain/lipid_library/lnpdb_to_heart.py" \
    "$REPO/model/pretrain/scripts/lipid_library/lnpdb_to_heart.py" \
    "$REPO/HEART_LNPDB_Adapter/lnpdb_to_heart.py"; do
    if [[ -f "$candidate" ]]; then
      ADAPTER="$candidate"
      break
    fi
  done
fi
if [[ -z "${ADAPTER:-}" ]]; then
  echo "ERROR: lnpdb_to_heart.py not found under known project locations." >&2
  exit 1
fi

for path in "$ADAPTER" "$PREPARED" "$DICT" "$PY" "$REPO/model/pretrain/lipid_library/conformers.py"; do
  if [[ ! -f "$path" ]]; then
    printf 'ERROR: Required file not found: %s\n' "$path" >&2
    exit 1
  fi
done
if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_SHARDS must be a positive integer." >&2
  exit 1
fi
if [[ ! -d "$OUT" ]]; then
  echo "ERROR: Output directory does not exist: $OUT" >&2
  exit 1
fi

# Do not finalize an incomplete array.
missing=0
for ((i=0;i<NUM_SHARDS;i++)); do
  printf -v shard 'shard-%05d' "$i"
  if [[ ! -f "$OUT/$shard/receipt.json" ]]; then
    echo "ERROR: Missing $OUT/$shard/receipt.json" >&2
    missing=1
  fi
done
if (( missing )); then
  echo 'ERROR: Some shards are incomplete. No finalize/verify attempted.' >&2
  exit 1
fi

REPORT_DIR="$ROOT/reports"
mkdir -p "$REPORT_DIR"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG="$REPORT_DIR/finalize_verify_${RUN_ID}.log"
SUMMARY="$REPORT_DIR/finalize_verify_${RUN_ID}_summary.json"

# Capture both stdout and stderr in a permanent log and on screen.
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; echo "[$(date -Is)] EXIT STATUS: $code"; echo "Log: '$LOG'"; exit "$code"' EXIT

echo "[$(date -Is)] HEART LNPDB finalize / full verification"
echo "Adapter: $ADAPTER"
echo "Dataset: $OUT"
echo "Prepared input: $PREPARED"
echo "Dictionary: $DICT"
echo "Shard count: $NUM_SHARDS"
echo "No conformer generation or Ugi modification will occur."

"$PY" "$ADAPTER" finalize \
  --out "$OUT" \
  --prepared "$PREPARED" \
  --num-shards "$NUM_SHARDS"

"$PY" "$ADAPTER" verify \
  --repo "$REPO" \
  --dict "$DICT" \
  --out "$OUT" \
  --check-all

"$PY" - "$OUT" "$SUMMARY" <<'PYCODE'
import json, pathlib, sys
root, dest = map(pathlib.Path, sys.argv[1:])
splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
verification = json.loads((root / 'verification.json').read_text(encoding='utf-8'))
receipts = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('shard-*/receipt.json'))]
summary = {
    'status': verification.get('status'),
    'complete': splits.get('complete'),
    'smoke': splits.get('smoke'),
    'n_shards': splits.get('n_shards'),
    'n_input_rows': splits.get('n_input_rows'),
    'n_records': splits.get('n_records'),
    'n_skipped': splits.get('n_skipped'),
    'n_train_unique': verification.get('n_train_unique'),
    'n_valid_unique': verification.get('n_valid_unique'),
    'coordinate_slots_per_record': splits.get('coordinate_slots_per_record'),
    'stored_2d_conformer': splits.get('stored_2d_conformer'),
    'geometry_counts': {},
    'skip_reasons': {},
    'verification_problems': verification.get('problems', []),
}
for r in receipts:
    for k, v in r.get('geometry', {}).items():
        if isinstance(v, (int, float)):
            summary['geometry_counts'][k] = summary['geometry_counts'].get(k, 0) + v
    for k, v in r.get('skip_reasons', {}).items():
        summary['skip_reasons'][k] = summary['skip_reasons'].get(k, 0) + v
dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('\n=== SUMMARY ===')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('\nPermanent summary: ' + str(dest))
if summary['status'] != 'PASS' or not summary['complete'] or summary['smoke']:
    sys.exit('ERROR: Dataset did not pass full production verification.')
PYCODE

echo "[$(date -Is)] FULL LNPDB VERIFICATION PASSED"
echo "Metadata: $OUT/splits.json"
echo "Validation: $OUT/verification.json"
echo "Summary: $SUMMARY"
