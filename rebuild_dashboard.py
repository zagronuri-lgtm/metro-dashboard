#!/usr/bin/env python3
"""
מטרופולין — Pipeline לבניית דשבורד אוטומטית
==============================================
שימוש:
  python3 rebuild_dashboard.py                                          # שימוש בנתונים קיימים
  python3 rebuild_dashboard.py --definitions "נובמבר 2025.csv"          # חודש אחד
  python3 rebuild_dashboard.py --definitions "אוק*.csv" "נוב*.csv"     # מספר חודשים
  python3 rebuild_dashboard.py --refresh-api                            # רענון נתוני data.gov.il
  python3 rebuild_dashboard.py --no-holidays                            # בלי סינון חגים

תפוקה:
  docs/index.html — דשבורד מוכן ל-GitHub Pages

סינון חגים:
  כברירת מחדל, נסיעות בחגים יהודיים, מוסלמיים ובשבתות מסוננות מהניתוח.
  כדי לבטל את הסינון: --no-holidays
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import glob as glob_mod

# Holiday filtering
from holidays import get_all_holidays

# === PATHS ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "docs"
DEFINITIONS_DIR = SCRIPT_DIR.parent / "route-definitions"

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# === CONFIG ===
TIKUFIM_RESOURCE_ID = "e72b10f3-4458-42c1-ba34-9b232feb8bc7"
BITZUA_RESOURCE_ID = "084b8e33-e359-47aa-95f7-26782e52c9af"
RIDERSHIP_FILE = DEFINITIONS_DIR / "ridership_clean.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ==============================================================================
# STEP 1: Load or refresh data.gov.il tikufim
# ==============================================================================
def fetch_tikufim(month=None):
    """Fetch Metropoline tikufim from data.gov.il API."""
    log("שולף תיקופי מסלקה מ-data.gov.il...")

    all_records = []
    offset = 0
    filters = {'operator_nm': 'מטרופולין'}
    if month:
        filters['month_key'] = month

    while True:
        params = {
            'resource_id': TIKUFIM_RESOURCE_ID,
            'filters': json.dumps(filters),
            'limit': '1000',
            'offset': str(offset)
        }
        url = 'https://data.gov.il/api/3/action/datastore_search?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            records = data['result']['records']
            all_records.extend(records)
            if len(records) < 1000:
                break
            offset += 1000

    log(f"  נשלפו {len(all_records):,} רשומות תיקופים")
    return all_records


def find_latest_tikufim_month():
    """Find the latest available month in tikufim data."""
    for month in range(12, 0, -1):
        params = {
            'resource_id': TIKUFIM_RESOURCE_ID,
            'filters': json.dumps({'operator_nm': 'מטרופולין', 'month_key': month}),
            'limit': '1'
        }
        url = 'https://data.gov.il/api/3/action/datastore_search?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            if data['result']['total'] > 0:
                return month
    return None


def refresh_api_data():
    """Refresh all data from data.gov.il APIs."""
    # Find latest month
    latest_month = find_latest_tikufim_month()
    log(f"חודש אחרון זמין בתיקופים: {latest_month}")

    # Fetch tikufim
    tikufim = fetch_tikufim(month=latest_month)
    tik_file = DATA_DIR / "tikufim_latest.json"
    with open(tik_file, 'w', encoding='utf-8') as f:
        json.dump({'month': latest_month, 'records': tikufim, 'fetched': datetime.now().isoformat()}, f, ensure_ascii=False)
    log(f"  נשמר ב-{tik_file}")

    # Fetch execution stats (non-execution rate)
    log("שולף נתוני תכנון מול ביצוע...")
    exec_stats = {}
    for m in [latest_month]:
        total_params = {
            'resource_id': BITZUA_RESOURCE_ID,
            'filters': json.dumps({'operator_nm': 'מטרופולין', 'trip_month': m}),
            'limit': '0'
        }
        url = 'https://data.gov.il/api/3/action/datastore_search?' + urllib.parse.urlencode(total_params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            total = data['result']['total']

        ne_params = {
            'resource_id': BITZUA_RESOURCE_ID,
            'filters': json.dumps({'operator_nm': 'מטרופולין', 'trip_month': m, 'erua_hachraga_ind': 1}),
            'limit': '0'
        }
        url2 = 'https://data.gov.il/api/3/action/datastore_search?' + urllib.parse.urlencode(ne_params)
        req2 = urllib.request.Request(url2)
        with urllib.request.urlopen(req2, timeout=30) as resp:
            data2 = json.loads(resp.read())
            non_exec = data2['result']['total']

        exec_stats[m] = {'total': total, 'non_exec': non_exec, 'rate': round(non_exec/total*100, 2) if total > 0 else 0}
        log(f"  חודש {m}: {total:,} נסיעות, {non_exec:,} אי-ביצוע ({exec_stats[m]['rate']}%)")

    exec_file = DATA_DIR / "execution_stats.json"
    with open(exec_file, 'w', encoding='utf-8') as f:
        json.dump({'stats': exec_stats, 'fetched': datetime.now().isoformat()}, f, ensure_ascii=False)

    return latest_month


# ==============================================================================
# STEP 2: Process definitions CSV
# ==============================================================================
def parse_duration(s):
    """Convert H:MM:SS or MM:SS to minutes."""
    if not s or s == '':
        return None
    parts = str(s).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    elif len(parts) == 2:
        return int(parts[0]) + int(parts[1]) / 60
    return None


def process_definitions(csv_paths, filter_holidays=True):
    """Process one or more Optibus definitions CSVs into profiles.

    csv_paths: list of Path objects
    filter_holidays: if True, removes rows on Jewish/Muslim holidays and Shabbat
    """
    all_rows = []
    holiday_dates = get_all_holidays() if filter_holidays else set()
    filtered_holiday = 0
    filtered_shabbat = 0

    for csv_path in csv_paths:
        log(f"טוען הגדרות: {csv_path.name}")
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            file_rows = list(reader)
        log(f"  {len(file_rows):,} רשומות נסיעה")

        if filter_holidays:
            before = len(file_rows)
            clean_rows = []
            for r in file_rows:
                date_str = r.get('TripSourceDate', '')
                weekday = r.get('יום בשבוע', '')

                # סינון שבתות
                if weekday == 'שבת':
                    filtered_shabbat += 1
                    continue

                # סינון חגים
                if date_str in holiday_dates:
                    filtered_holiday += 1
                    continue

                clean_rows.append(r)

            log(f"  סינון: {before - len(clean_rows):,} שורות הוסרו ({filtered_shabbat:,} שבתות, {filtered_holiday:,} חגים)")
            file_rows = clean_rows

        all_rows.extend(file_rows)

    rows = all_rows
    log(f"סה\"כ: {len(rows):,} רשומות נסיעה ({len(csv_paths)} קבצים)")

    # Detect column names (handle variations)
    sample = rows[0]
    cols = list(sample.keys())

    # Find relevant columns
    line_col = next((c for c in cols if 'RouteShortName' in c or 'route_short' in c.lower()), None)
    dir_col = next((c for c in cols if c in ('Direction', 'direction', 'כיוון')), None)
    hour_col = next((c for c in cols if 'DataSourceHour' in c or 'Hour' in c or 'hour' in c), None)
    planned_col = next((c for c in cols if 'PlannedDuration' in c or 'planned' in c.lower()), None)
    duration_col = next((c for c in cols if c == 'Duration' or c == 'duration'), None)
    q85_col = next((c for c in cols if '85' in c), None)
    q90_col = next((c for c in cols if '90' in c), None)
    cluster_col = next((c for c in cols if 'Cluster' in c or 'cluster' in c), None)

    # Q85 might be missing in older files — we'll compute from raw durations
    compute_percentiles = q85_col is None

    if not all([line_col, dir_col, planned_col]):
        log(f"  ERROR: חסרות עמודות חובה. נמצאו: {cols}")
        sys.exit(1)

    if compute_percentiles:
        log(f"  עמודות: line={line_col}, dir={dir_col}, hour={hour_col}, planned={planned_col}")
        log(f"  אחוזונים חסרים — יחושבו מנתוני משך נסיעה גולמיים")
    else:
        log(f"  עמודות: line={line_col}, dir={dir_col}, hour={hour_col}, planned={planned_col}, q85={q85_col}")

    # Cluster name mapping
    CLUSTER_MAP = {
        'שרון חולון מרחבי': 'שרון-חולון',
        'בקעת אונו אלעד': 'אונו-אלעד',
        'שרון': 'שרון עירוני',
        'הנגב': 'נגב',
    }

    # Group by line+dir+hour
    groups = defaultdict(list)
    for r in rows:
        line = r.get(line_col, '').strip()
        direction = r.get(dir_col, '').strip()
        hour_raw = r.get(hour_col, '0').strip() if hour_col else '0'
        # Hour may be "HH:MM" or just a number
        if ':' in hour_raw:
            hour = hour_raw.split(':')[0]
        else:
            hour = hour_raw

        planned = parse_duration(r.get(planned_col, ''))
        actual = parse_duration(r.get(duration_col, '')) if duration_col else None
        cluster = r.get(cluster_col, '') if cluster_col else ''

        if compute_percentiles:
            q85 = None  # will compute later
            q90 = None
            if planned is None or actual is None:
                continue
        else:
            q85 = parse_duration(r.get(q85_col, ''))
            q90 = parse_duration(r.get(q90_col, '')) if q90_col else None
            if planned is None or q85 is None:
                continue

        try:
            key = (int(float(line)), int(float(direction)), int(float(hour)))
        except (ValueError, TypeError):
            continue

        groups[key].append({
            'planned': planned,
            'q85': q85,
            'q90': q90,
            'actual': actual,
            'cluster': cluster
        })

    # Aggregate to profiles
    def percentile(values, pct):
        """Compute percentile from a list of values."""
        if not values:
            return 0
        s = sorted(values)
        k = (len(s) - 1) * pct / 100
        f = int(k)
        c = f + 1 if f + 1 < len(s) else f
        return s[f] + (k - f) * (s[c] - s[f])

    profiles = []
    for (line, direction, hour), trips in groups.items():
        n = len(trips)
        avg_planned = sum(t['planned'] for t in trips) / n

        if compute_percentiles:
            # Compute percentiles from raw duration data
            durations = [t['actual'] for t in trips if t['actual'] is not None]
            if not durations:
                continue
            avg_q85 = percentile(durations, 85)
            avg_q90 = percentile(durations, 90)
        else:
            avg_q85 = sum(t['q85'] for t in trips) / n
            avg_q90 = sum(t['q90'] for t in trips if t['q90'] is not None) / max(1, sum(1 for t in trips if t['q90'] is not None))

        avg_actual = sum(t['actual'] for t in trips if t['actual'] is not None) / max(1, sum(1 for t in trips if t['actual'] is not None))

        gap = avg_q85 - avg_planned
        if compute_percentiles:
            pct_over = sum(1 for t in trips if t['actual'] is not None and t['actual'] > t['planned']) / n * 100
        else:
            pct_over = sum(1 for t in trips if t['q85'] is not None and t['q85'] > t['planned']) / n * 100

        cluster = trips[0]['cluster']
        branch = CLUSTER_MAP.get(cluster, cluster)

        profiles.append({
            'line': line,
            'dir': direction,
            'hour': hour,
            'branch': branch,
            'planned': round(avg_planned, 1),
            'actual': round(avg_actual, 1),
            'q85': round(avg_q85, 1),
            'q90': round(avg_q90, 1),
            'gap_q85': round(gap, 1),
            'n_trips': n,
            'pct_over': round(pct_over, 1)
        })

    log(f"  {len(profiles):,} פרופילים ייחודיים")
    return profiles


# ==============================================================================
# STEP 3: Enrich with ridership
# ==============================================================================
def safe_float(v, default=0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def load_ridership():
    """Load ridership_clean.csv."""
    if not RIDERSHIP_FILE.exists():
        log(f"  WARNING: {RIDERSHIP_FILE} לא נמצא")
        return {}

    with open(RIDERSHIP_FILE, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    # Dedup by line+dir, prefer latest quarter
    rid_by_q = defaultdict(lambda: defaultdict(lambda: {'daily': 0, 'weekly': 0, 'per_ride': 0}))
    for r in rows:
        key = f'{r["RouteName"]}_{r["RouteDirection"]}'
        q = int(r.get('Q', 0))
        rid_by_q[key][q]['daily'] += safe_float(r.get('DailyPassengers'))
        rid_by_q[key][q]['weekly'] += safe_float(r.get('WeeklyPassengers'))
        rid_by_q[key][q]['per_ride'] = max(rid_by_q[key][q]['per_ride'], safe_float(r.get('AVGCommutersPerRide(Weekly)')))

    lookup = {}
    for key, quarters in rid_by_q.items():
        best_q = max(quarters.keys())
        lookup[key] = quarters[best_q]

    log(f"  טעון ridership: {len(lookup)} צירופי קו×כיוון")
    return lookup


def enrich_profiles(profiles, ridership):
    """Add ridership data to profiles."""
    matched = 0
    for p in profiles:
        key = f'{p["line"]}_{p["dir"]}'
        rid = ridership.get(key)
        if rid:
            p['daily_pax'] = round(rid['daily'], 1)
            p['pax_per_ride'] = round(rid['per_ride'], 1)
            matched += 1
        else:
            p['daily_pax'] = 0
            p['pax_per_ride'] = 0

        gap = max(p['gap_q85'], 0)
        p['priority'] = round(gap * p['daily_pax'])
        p['impact_per_ride'] = round(gap * p['pax_per_ride'], 1)

    if len(profiles) > 0:
        log(f"  הצלבת ridership: {matched}/{len(profiles)} ({matched/len(profiles)*100:.1f}%)")
    else:
        log("  WARNING: אין פרופילים להצלבה")
    return profiles


# ==============================================================================
# STEP 4: Build aggregated stats
# ==============================================================================
def build_aggregates(profiles):
    """Build branch stats, hour stats, priority lines."""

    # Branch stats
    branch_map = defaultdict(lambda: {'gap_sum': 0, 'count': 0})
    for p in profiles:
        branch_map[p['branch']]['gap_sum'] += p['gap_q85']
        branch_map[p['branch']]['count'] += 1
    branch_stats = [{'branch': b, 'avg_gap': round(v['gap_sum']/v['count'], 2), 'count': v['count']} for b, v in branch_map.items()]

    # Hour stats
    hour_map = defaultdict(lambda: {'gap_sum': 0, 'count': 0})
    for p in profiles:
        hour_map[p['hour']]['gap_sum'] += p['gap_q85']
        hour_map[p['hour']]['count'] += 1
    hour_stats = [{'hour': h, 'avg_gap': round(v['gap_sum']/v['count'], 2), 'count': v['count']} for h, v in hour_map.items()]

    # Priority lines
    line_agg = defaultdict(lambda: {
        'line': 0, 'dir': 0, 'branch': '', 'gap_sum': 0, 'gap_count': 0,
        'max_gap': 0, 'daily_pax': 0, 'pax_per_ride': 0, 'bad_hours': 0, 'profiles': 0
    })
    for p in profiles:
        key = f'{p["line"]}_{p["dir"]}'
        lp = line_agg[key]
        lp['line'] = p['line']; lp['dir'] = p['dir']; lp['branch'] = p['branch']
        lp['daily_pax'] = p['daily_pax']; lp['pax_per_ride'] = p['pax_per_ride']
        lp['profiles'] += 1
        if p['gap_q85'] > 0:
            lp['gap_sum'] += p['gap_q85']; lp['gap_count'] += 1
            lp['max_gap'] = max(lp['max_gap'], p['gap_q85'])
        if p['gap_q85'] > 5:
            lp['bad_hours'] += 1

    priority_list = []
    for v in line_agg.values():
        avg_gap = v['gap_sum'] / v['gap_count'] if v['gap_count'] > 0 else 0
        priority_list.append({
            'line': v['line'], 'dir': v['dir'], 'branch': v['branch'],
            'avg_gap': round(avg_gap, 1), 'max_gap': round(v['max_gap'], 1),
            'bad_hours': v['bad_hours'], 'daily_pax': v['daily_pax'],
            'pax_per_ride': v['pax_per_ride'],
            'priority': round(avg_gap * v['daily_pax']),
            'impact_per_ride': round(avg_gap * v['pax_per_ride'], 1),
            'profiles': v['profiles']
        })
    priority_list.sort(key=lambda x: x['priority'], reverse=True)

    # KPIs
    n = len(profiles)
    over_q85 = sum(1 for p in profiles if p['gap_q85'] > 0)
    over_5 = sum(1 for p in profiles if p['gap_q85'] > 5)
    avg_gap = sum(p['gap_q85'] for p in profiles) / n if n > 0 else 0

    total_daily_pax = 0
    seen_ld = set()
    for p in profiles:
        k = f'{p["line"]}_{p["dir"]}'
        if k not in seen_ld:
            seen_ld.add(k)
            total_daily_pax += p.get('daily_pax', 0)

    kpis = {
        'total_profiles': n,
        'total_lines': len(set(p['line'] for p in profiles)),
        'pct_over_q85': round(over_q85 / n * 100, 1) if n > 0 else 0,
        'pct_over_5': round(over_5 / n * 100, 1) if n > 0 else 0,
        'avg_gap': round(avg_gap, 1),
        'total_daily_pax': round(total_daily_pax),
        'total_std': 40676  # from STD analysis
    }

    return {
        'branch_stats': branch_stats,
        'hour_stats': hour_stats,
        'priority_lines': priority_list[:60],
        'kpis': kpis
    }


# ==============================================================================
# STEP 5: Load STD data (cached from previous analysis)
# ==============================================================================
def load_std_data():
    """Load STD Top 500 from cache."""
    std_file = DATA_DIR / "std_top500.json"
    if std_file.exists():
        with open(std_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Try loading from compact_nov.json
    compact_file = Path("/sessions/admiring-funny-ride/compact_nov.json")
    if compact_file.exists():
        with open(compact_file, 'r') as f:
            compact = json.load(f)
        std = compact.get('std_top500', [])
        # Cache it
        with open(std_file, 'w', encoding='utf-8') as f:
            json.dump(std, f, ensure_ascii=False)
        return std

    log("  WARNING: נתוני STD לא נמצאו")
    return []


# ==============================================================================
# STEP 6: Build HTML dashboard
# ==============================================================================
def build_html(profiles, aggregates, std_data, definitions_source, data_date):
    """Build the complete HTML dashboard."""
    log("בונה דשבורד HTML...")

    # Read the template from build_dashboard_v4.py output structure
    # We'll embed data directly

    setup_json = json.dumps(profiles, ensure_ascii=False)
    std_json = json.dumps(std_data, ensure_ascii=False)
    branch_json = json.dumps(aggregates['branch_stats'], ensure_ascii=False)
    hour_json = json.dumps(aggregates['hour_stats'], ensure_ascii=False)
    kpis_json = json.dumps(aggregates['kpis'], ensure_ascii=False)
    priority_json = json.dumps(aggregates['priority_lines'], ensure_ascii=False)

    now = datetime.now().strftime('%d/%m/%Y %H:%M')

    html_parts = []

    # We'll read the template from an external file if it exists,
    # otherwise use the embedded one
    template_file = SCRIPT_DIR / "dashboard_template.html"

    if template_file.exists():
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()

        # Replace data placeholders
        template = template.replace('/*DATA_SETUP*/', f'const SETUP_ALL = {setup_json};')
        template = template.replace('/*DATA_STD*/', f'const STD_TOP500 = {std_json};')
        template = template.replace('/*DATA_BRANCH*/', f'const BRANCH_STATS = {branch_json};')
        template = template.replace('/*DATA_HOUR*/', f'const HOUR_STATS = {hour_json};')
        template = template.replace('/*DATA_KPIS*/', f'const KPIS = {kpis_json};')
        template = template.replace('/*DATA_PRIORITY*/', f'const PRIORITY_LINES = {priority_json};')
        template = template.replace('{{BUILD_DATE}}', now)
        template = template.replace('{{DATA_DATE}}', data_date)
        template = template.replace('{{DEFINITIONS_SOURCE}}', definitions_source)

        html_parts.append(template)
    else:
        # Fallback: copy from current dashboard and inject data
        log("  Template לא נמצא, משתמש ב-metro_dashboard.html הנוכחי כבסיס")
        current = SCRIPT_DIR.parent / "metro_dashboard.html"
        if not current.exists():
            current = Path("/sessions/admiring-funny-ride/mnt/uriz/metro_dashboard.html")

        with open(current, 'r', encoding='utf-8') as f:
            html = f.read()

        # Replace data blocks
        import re
        html = re.sub(r'const SETUP_ALL = .+?;', f'const SETUP_ALL = {setup_json};', html, count=1, flags=re.DOTALL)
        html = re.sub(r'const STD_TOP500 = .+?;', f'const STD_TOP500 = {std_json};', html, count=1, flags=re.DOTALL)
        html = re.sub(r'const BRANCH_STATS = .+?;', f'const BRANCH_STATS = {branch_json};', html, count=1, flags=re.DOTALL)
        html = re.sub(r'const HOUR_STATS = .+?;', f'const HOUR_STATS = {hour_json};', html, count=1, flags=re.DOTALL)
        html = re.sub(r'const KPIS = .+?;', f'const KPIS = {kpis_json};', html, count=1, flags=re.DOTALL)
        html = re.sub(r'const PRIORITY_LINES = .+?;', f'const PRIORITY_LINES = {priority_json};', html, count=1, flags=re.DOTALL)

        html_parts.append(html)

    output = ''.join(html_parts)

    # Write output
    output_file = OUTPUT_DIR / "index.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

    log(f"  דשבורד נכתב: {output_file} ({len(output)/1024:.0f} KB)")
    return output_file


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='מטרופולין — בניית דשבורד')
    parser.add_argument('--definitions', '-d', nargs='+', help='קובץ/קבצי הגדרות CSV מ-Optibus (אפשר כמה)')
    parser.add_argument('--refresh-api', '-r', action='store_true', help='רענון נתונים מ-data.gov.il')
    parser.add_argument('--no-holidays', action='store_true', help='בלי סינון חגים ושבתות')
    parser.add_argument('--output', '-o', help='נתיב קובץ פלט (ברירת מחדל: docs/index.html)')
    args = parser.parse_args()

    log("=" * 60)
    log("מטרופולין — Pipeline בניית דשבורד")
    log("=" * 60)

    # Step 1: API data
    if args.refresh_api:
        latest_month = refresh_api_data()
        data_date = f"חודש {latest_month}/2025"
    else:
        data_date = "נובמבר 2025"

    # Step 2: Definitions
    filter_holidays = not args.no_holidays
    if filter_holidays:
        log("סינון חגים ושבתות: מופעל")
    else:
        log("סינון חגים ושבתות: כבוי (--no-holidays)")

    if args.definitions:
        # Resolve paths — support glob patterns and multiple files
        csv_paths = []
        for pattern in args.definitions:
            p = Path(pattern)
            if not p.is_absolute():
                p = DEFINITIONS_DIR / pattern
            # Try glob expansion
            matches = sorted(glob_mod.glob(str(p)))
            if matches:
                csv_paths.extend(Path(m) for m in matches)
            elif p.exists():
                csv_paths.append(p)
            else:
                log(f"WARNING: לא נמצא קובץ: {pattern}")

        if not csv_paths:
            log("ERROR: לא נמצאו קבצי הגדרות")
            sys.exit(1)

        log(f"נמצאו {len(csv_paths)} קבצי הגדרות:")
        for cp in csv_paths:
            log(f"  — {cp.name}")

        profiles = process_definitions(csv_paths, filter_holidays=filter_holidays)
        definitions_source = ', '.join(cp.name for cp in csv_paths)

        # Cache processed profiles
        cache_file = DATA_DIR / "profiles_latest.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False)
        log(f"  פרופילים נשמרו ב-{cache_file}")
    else:
        # Load cached profiles
        cache_file = DATA_DIR / "profiles_latest.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
            definitions_source = "מטמון"
            log(f"טעון פרופילים מ-cache: {len(profiles):,}")
        else:
            # Try to find latest CSV in definitions dir
            csvs = sorted(DEFINITIONS_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True)
            if csvs:
                log(f"נמצא CSV עדכני: {csvs[0].name}")
                profiles = process_definitions([csvs[0]], filter_holidays=filter_holidays)
                definitions_source = csvs[0].name
            else:
                log("ERROR: לא נמצאו קבצי הגדרות ולא cache")
                sys.exit(1)

    # Step 3: Ridership
    ridership = load_ridership()
    profiles = enrich_profiles(profiles, ridership)

    # Step 4: Aggregates
    aggregates = build_aggregates(profiles)

    # Step 5: STD
    std_data = load_std_data()

    # Step 6: Build HTML
    output_file = build_html(profiles, aggregates, std_data, definitions_source, data_date)

    if args.output:
        import shutil
        shutil.copy(output_file, args.output)
        log(f"  הועתק גם ל-{args.output}")

    log("")
    log("=" * 60)
    log(f"✅ הדשבורד מוכן: {output_file}")
    log(f"   פרופילים: {len(profiles):,}")
    log(f"   הגדרות: {definitions_source}")
    log(f"   נתונים: {data_date}")
    log("=" * 60)


if __name__ == '__main__':
    main()
