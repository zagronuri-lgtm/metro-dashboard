#!/bin/bash
# עדכון הדשבורד עם הגדרות חדשות
# שימוש: ./update.sh "שם_הקובץ.csv"
# או פשוט: ./update.sh (ייקח את הקובץ החדש ביותר אוטומטית)

cd "$(dirname "$0")"

echo "============================="
echo "  עדכון דשבורד מטרופולין"
echo "============================="
echo ""

if [ -n "$1" ]; then
    CSV="$1"
    echo "קובץ הגדרות: $CSV"
    python3 rebuild_dashboard.py --definitions "$CSV" --refresh-api
else
    echo "לא צוין קובץ — משתמש בנתונים קיימים + רענון נתוני משרד התחבורה"
    python3 rebuild_dashboard.py --refresh-api
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "שגיאה בבניית הדשבורד!"
    exit 1
fi

echo ""
echo "מעלה לאתר..."
git add .
git commit -m "עדכון: $(date +%d/%m/%Y)"
git push

echo ""
echo "============================="
echo "  הדשבורד עודכן בהצלחה!"
echo "============================="
