#!/usr/bin/env bash
# End-to-end demo: submit an analysis, poll until done, print the report.
set -euo pipefail
BASE="${BASE_URL:-http://localhost:8000}"
QUERY="${1:-iPhone 16}"

echo "── Health ──────────────────────────────"
curl -s "$BASE/health" | python3 -m json.tool

echo "── Submitting analysis for: $QUERY ─────"
ID=$(curl -s -X POST "$BASE/api/v1/analyses" \
  -H 'Content-Type: application/json' \
  -d "{\"product\": \"$QUERY\"}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "analysis id: $ID"

echo "── Progress (polling) ──────────────────"
for _ in $(seq 1 60); do
  STATUS=$(curl -s "$BASE/api/v1/analyses/$ID" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "status: $STATUS"
  [[ "$STATUS" == "done" || "$STATUS" == "failed" ]] && break
  sleep 0.5
done

echo "── Report (markdown) ───────────────────"
curl -s "$BASE/api/v1/analyses/$ID/report.md"

echo "── Metadata ────────────────────────────"
curl -s "$BASE/api/v1/analyses/$ID" | python3 -c \
  'import json,sys; print(json.dumps(json.load(sys.stdin)["meta"], indent=2))'
