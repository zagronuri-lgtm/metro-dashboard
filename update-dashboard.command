#!/bin/bash
cd "$(dirname "$0")"

DEFS_DIR="../הגדרות קווים  אחוזונים/ביצוע נסיעות"

echo "=============================="
echo "  Metro Dashboard - Update"
echo "=============================="
echo ""

# Find 2 most recent CSV files
FILES=($(ls -1t "$DEFS_DIR"/*.csv 2>/dev/null | head -2))

if [ ${#FILES[@]} -eq 0 ]; then
  echo "ERROR: No CSV files found"
  echo "Press Enter to close..."
  read
  exit 1
fi

echo "Selected definitions:"
for f in "${FILES[@]}"; do echo "  > $(basename "$f")"; done
echo ""
echo "Continue? (Enter = yes, n = no)"
read -r answer
if [ "$answer" = "n" ]; then
  echo "Cancelled."
  read
  exit 0
fi

echo ""
python3 rebuild_dashboard.py --refresh-api --definitions "${FILES[@]}"

echo ""
echo "Done! Dashboard updated."
echo "Press Enter to close..."
read
