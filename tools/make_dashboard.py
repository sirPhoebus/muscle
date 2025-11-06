from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from datetime import datetime


def find_latest_csv(stats_dir: Path) -> Path | None:
    if not stats_dir.exists():
        return None
    csvs = sorted(stats_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def read_series(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        keys = reader.fieldnames or []
        series: dict[str, list[float]] = {k: [] for k in keys}
        for row in reader:
            for k in keys:
                v = row.get(k)
                if v is None:
                    continue
                try:
                    # cast to float when possible
                    series[k].append(float(v))
                except ValueError:
                    series[k].append(v)  # keep raw
    return series


def write_html(out_path: Path, series: dict[str, list[float]], title: str) -> None:
    # Minimal HTML + Chart.js dashboard (no Python deps required)
    data_json = json.dumps(series)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js\"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f1a; color:#e6e9f2; margin:0; }}
    .wrap {{ max-width:1200px; margin: 24px auto; padding: 0 16px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }}
    .card {{ background:#121a2a; border:1px solid #2a3a5a; border-radius:12px; padding:14px; }}
    h1 {{ font-size:20px; margin: 0 0 12px; color:#eaf0ff; }}
    h2 {{ font-size:16px; margin: 0 0 10px; color:#d9e4ff; }}
    .muted {{ color:#a8b3cf; font-size:13px; }}
    .kpi {{ font-size:28px; font-weight:600; margin:6px 0 2px; }}
    canvas {{ background:#0c1322; border-radius:8px; }}
  </style>
  <script>
    const SERIES = {data_json};
    function last(arr, defv=0) {{ return (arr && arr.length)? arr[arr.length-1]: defv; }}
    function ready() {{
      const t = SERIES.t || [];
      const pop = SERIES.pop || [];
      const readyArr = SERIES.ready || [];
      const births = SERIES.births || [];
      const deaths = SERIES.deaths || [];
      const foodRatio = SERIES.food_ratio || [];
      const avgSpd = SERIES.avg_spd || [];
      const maxSpd = SERIES.max_spd || [];

      // KPIs
      document.getElementById('kpi-time').textContent = (last(t,0)).toFixed(1) + ' s';
      document.getElementById('kpi-pop').textContent = (last(pop,0)).toFixed(0);
      document.getElementById('kpi-births').textContent = (last(births,0)).toFixed(0);
      document.getElementById('kpi-deaths').textContent = (last(deaths,0)).toFixed(0);
      document.getElementById('kpi-food').textContent = ((last(foodRatio,0))*100).toFixed(0) + '%';

      const commonOpts = {{
        responsive: true,
        scales: {{
          x: {{ ticks: {{ color:'#9fb2d9' }}, grid: {{ color:'rgba(159,178,217,0.15)'}} }},
          y: {{ ticks: {{ color:'#9fb2d9' }}, grid: {{ color:'rgba(159,178,217,0.15)'}} }}
        }},
        plugins: {{ legend: {{ labels: {{ color:'#dfe8ff' }} }} }}
      }};

      new Chart(document.getElementById('ch-pop'), {{
        type:'line', data: {{ labels: t, datasets: [
          {{ label:'Population', data: pop, borderColor:'#66ccff', tension:0.15 }},
          {{ label:'Ready', data: readyArr, borderColor:'#a7f3d0', tension:0.15 }}
        ]}}, options: commonOpts
      }});

      new Chart(document.getElementById('ch-food'), {{
        type:'line', data: {{ labels: t, datasets: [
          {{ label:'Food Ratio', data: foodRatio, borderColor:'#fbbf24', tension:0.15 }}
        ]}}, options: commonOpts
      }});

      new Chart(document.getElementById('ch-speed'), {{
        type:'line', data: {{ labels: t, datasets: [
          {{ label:'Avg Speed', data: avgSpd, borderColor:'#a78bfa', tension:0.15 }},
          {{ label:'Max Speed', data: maxSpd, borderColor:'#f472b6', tension:0.15 }}
        ]}}, options: commonOpts
      }});
    }}
    window.addEventListener('DOMContentLoaded', ready);
  </script>
</head>
<body>
  <div class=\"wrap\">
    <h1>{title}</h1>
    <div class=\"muted\">Generated {now}</div>
    <div class=\"grid\" style=\"margin-top:12px\">
      <div class=\"card\"><h2>Time</h2><div id=\"kpi-time\" class=\"kpi\">–</div></div>
      <div class=\"card\"><h2>Population</h2><div id=\"kpi-pop\" class=\"kpi\">–</div></div>
      <div class=\"card\"><h2>Births</h2><div id=\"kpi-births\" class=\"kpi\">–</div></div>
      <div class=\"card\"><h2>Deaths</h2><div id=\"kpi-deaths\" class=\"kpi\">–</div></div>
      <div class=\"card\"><h2>Food Availability</h2><div id=\"kpi-food\" class=\"kpi\">–</div></div>
    </div>
    <div class=\"grid\" style=\"margin-top:16px\">
      <div class=\"card\"><h2>Population / Ready</h2><canvas id=\"ch-pop\" height=\"180\"></canvas></div>
      <div class=\"card\"><h2>Food Ratio</h2><canvas id=\"ch-food\" height=\"180\"></canvas></div>
      <div class=\"card\"><h2>Speed</h2><canvas id=\"ch-speed\" height=\"180\"></canvas></div>
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    # Usage: python tools/make_dashboard.py [stats_dir]
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    stats_dir = (base / 'stats').resolve()
    latest = find_latest_csv(stats_dir)
    if latest is None:
        print(f"No CSV files found in {stats_dir}")
        sys.exit(1)
    series = read_series(latest)
    out_path = stats_dir / 'dashboard_latest.html'
    title = f"Worm Simulation Dashboard — {latest.name}"
    write_html(out_path, series, title)
    print(f"Dashboard written to: {out_path}")


if __name__ == '__main__':
    main()

