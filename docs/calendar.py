# scrape yaml files to create summaries for last couple days pushes for Readme.md

import datetime
import yaml
import json
import os
from pathlib import Path

# index.html reads the day folder out of "day" by splitting on backslashes, so the
# field is always written in this shape no matter which machine generates the file
WIN_PREFIX = "C:\\Users\\amkah\\OneDrive\\Documents\\GitHub\\sandbox"

today = datetime.date.today()
month_name = today.strftime("%B").lower()
if os.environ.get("GITHUB_ACTIONS"):
    base_dir = Path("/home/runner/work/sandbox/sandbox").resolve()
else:
    base_dir = Path(__file__).parent.parent.resolve()

# every month folder in the repo root, e.g. 08_august ... 12_december, 01_january
months = sorted(p for p in base_dir.glob("[0-9][0-9]_*") if p.is_dir())

print(f"\nbase dir: {base_dir}")
for month in months:
    print(f"{month.name} exists: {month.exists()}")

day_folders = []
for month_dir in months:
    for folder in month_dir.iterdir():
        if folder.is_dir():
            try:
                month, day, year = folder.name.split("_")
                date = datetime.date(int("20" + year), int(month), int(day))

                notes_file = folder / "notes.yaml"
                if notes_file.exists() and notes_file.stat().st_size > 0:
                    day_folders.append((date, folder))

            except Exception:
                continue

notes_content = ""
day_folders.sort(key=lambda x: x[0])
previous_days = [f for d, f in reversed(day_folders) if d <= today]
for folder in previous_days:
    notes_file = folder / "notes.yaml"
    with open(notes_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    repo_day = f"{folder.parent.name}/{folder.name}"
    data["day"] = f"{WIN_PREFIX}\\{folder.parent.name}\\{folder.name}\\notes.yaml"
    data["repo_url"] = f"https://github.com/AmiriHayes/sandbox/tree/main/{repo_day}"
    notes_content += str(json.dumps(data))
    notes_content += "\n"

research_notes_file = base_dir / "docs/research.jsonl"
with open(research_notes_file, "r", encoding="utf-8") as f:
    research_data = list(f)
for json_str in research_data:
    # lines are written with a trailing comma, which may or may not be newline terminated
    research_day = json_str.strip().rstrip(",")
    if not research_day:
        continue
    research_focused_day = json.loads(research_day)

    day = research_focused_day.get("day")
    day_formatted = f"{WIN_PREFIX}\\{day[:2]}_{day[3:5]}_{day[6:]}\\notes.yaml"
    project = research_focused_day.get("project")
    notes_content += str(json.dumps({
        "day": day_formatted,
        "type": "research",
        "project": project,})) + "\n"

output_file = base_dir / "docs/calendar.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(notes_content)

print(f"Wrote formatted notes from {len(previous_days)+len(research_data)} non-empty days up to {month_name} to {output_file}")
