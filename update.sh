#!/bin/bash
# עדכון הדשבורד עם הגדרות חדשות
# שימוש: ./update.sh "שם_הקובץ.csv"
# או פשוט: ./update.sh (ייקח את הקובץ החדש ביותר אוטומטית)

cd "$(dirname "$0")"
ROUTE_DEFS="../הגדרות קווים  אחוזונים/ביצוע נסיעות"

echo "============================="
echo "  עדכון דשבורד מטרופולין"
echo "============================="
echo ""

# סנכרון תיקיית route-definitions → definitions/
if [ -d "$ROUTE_DEFS" ]; then
    mkdir -p definitions
    NEW_FILES=0
    for csv in "$ROUTE_DEFS"/*.csv; do
        [ -f "$csv" ] || continue
        BASENAME="$(basename "$csv")"
        if [ ! -f "definitions/$BASENAME" ] || [ "$csv" -nt "definitions/$BASENAME" ]; then
            cp "$csv" "definitions/"
            echo "סונכרן: $BASENAME"
            NEW_FILES=$((NEW_FILES + 1))
        fi
    done
    if [ $NEW_FILES -eq 0 ]; then
        echo "definitions/ מעודכנת — אין קבצים חדשים"
    else
        echo "סונכרנו $NEW_FILES קבצים לתיקיית definitions/"
    fi
    echo ""
fi

if [ -n "$1" ]; then
    # קבצים שצוינו ידנית
    echo "קבצי הגדרות: $@"
    python3 rebuild_dashboard.py --definitions "$@" --refresh-api
else
    # לקיחת 3 הקבצים האחרונים לפי תאריך שינוי
    LATEST=$(ls -t "$ROUTE_DEFS"/*.csv 2>/dev/null | head -3)
    if [ -n "$LATEST" ]; then
        echo "3 קבצי ההגדרות האחרונים:"
        for f in $LATEST; do echo "  — $(basename "$f")"; done
        echo ""
        python3 rebuild_dashboard.py --definitions $LATEST --refresh-api
    else
        echo "לא נמצאו קבצי הגדרות — משתמש בנתונים קיימים"
        python3 rebuild_dashboard.py --refresh-api
    fi
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
