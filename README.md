# מטרופולין — Dashboard Pipeline

כלי אוטומטי לבניית דשבורד ניתוח דיוק הגדרות מול ביצוע + נפח נוסעים.

## מבנה הפרויקט

```
dashboard-pipeline/
├── rebuild_dashboard.py      # סקריפט הבנייה הראשי
├── dashboard_template.html   # תבנית הדשבורד
├── data/                     # נתונים מוכנים (cache)
│   ├── profiles_latest.json  # פרופילים אחרונים מעובדים
│   ├── std_top500.json       # TOP 500 פרופילי STD
│   ├── tikufim_latest.json   # תיקופי מסלקה (אחרי refresh)
│   └── execution_stats.json  # נתוני ביצוע (אחרי refresh)
├── docs/
│   └── index.html            # הדשבורד (GitHub Pages)
├── .github/workflows/
│   └── refresh.yml           # עדכון אוטומטי שבועי
└── README.md
```

## שימוש

### עדכון הגדרות (ידני — כל חודש)

כשמתקבל קובץ הגדרות חדש מ-Optibus, שמור אותו בתיקיית `route-definitions` והרץ:

```bash
python3 rebuild_dashboard.py --definitions "דצמבר 2025.csv"
```

### עדכון נתוני משרד התחבורה (אוטומטי / ידני)

```bash
python3 rebuild_dashboard.py --refresh-api
```

### שילוב שניהם

```bash
python3 rebuild_dashboard.py --definitions "דצמבר 2025.csv" --refresh-api
```

### שימוש ב-cache בלבד (בלי שינויים)

```bash
python3 rebuild_dashboard.py
```

## GitHub Pages

הדשבורד נבנה לתוך `docs/index.html`. כדי להפעיל GitHub Pages:

1. צור repository חדש ב-GitHub
2. העלה את כל הקבצים:
   ```bash
   cd dashboard-pipeline
   git init
   git add .
   git commit -m "Initial: dashboard pipeline"
   git remote add origin https://github.com/USERNAME/metro-dashboard.git
   git push -u origin main
   ```
3. ב-GitHub → Settings → Pages → Source: `main` / `docs`
4. הדשבורד יהיה זמין ב: `https://USERNAME.github.io/metro-dashboard/`

## עדכון אוטומטי

GitHub Actions מריץ `--refresh-api` כל יום ראשון ב-06:00 (שעון ישראל).
אפשר גם להריץ ידנית: Actions → Refresh Dashboard Data → Run workflow.

## מקורות נתונים

| מקור | סוג | עדכון |
|------|------|-------|
| data.gov.il — תיקופי מסלקה | API אוטומטי | שבועי |
| data.gov.il — תכנון מול ביצוע | API אוטומטי | שבועי |
| data.gov.il — STD 2024 | cache סטטי | שנתי |
| Optibus — הגדרות | CSV ידני | חודשי |
| ridership_clean.csv | CSV ידני | רבעוני |
