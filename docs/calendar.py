# scrape yaml files to create summaries for last couple days pushes for Readme.md

import datetime
import yaml
import json
import os
from pathlib import Path

today = datetime.date.today()
month_name = today.strftime("%B").lower()
months = []
if os.environ.get("GITHUB_ACTIONS"):
    base_dir = Path("/home/runner/work/sandbox/sandbox").resolve()
    august = Path("/home/runner/work/sandbox/sandbox/08_august")
    september = Path("/home/runner/work/sandbox/sandbox/09_september")
    months.extend((august, september))
else:
    base_dir = Path("../sandbox").resolve()
    august = base_dir / "08_august"
    september = base_dir / "09_september"
    months.extend((august, september))

print(f"\nbase dir: {base_dir}")
print(f"august dir: {august.resolve()}")
print(f"september dir: {september.resolve()}\n")

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
    print(f"type data: {type(data)}")
    data["day"] = str(notes_file)
    repo_day = str(notes_file.parent.relative_to(notes_file.parents[2]))
    data["repo_url"] = f"https://github.com/AmiriHayes/sandbox/tree/main/{repo_day}"
    notes_content += str(json.dumps(data))
    notes_content += "\n"

if os.environ.get("GITHUB_ACTIONS"):
    output_file = Path("/home/runner/work/sandbox/sandbox/docs/calendar_2.jsonl")
else:
    project_root = Path(__file__).parent.parent.resolve()
    output_file = project_root / "docs/calendar.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(notes_content)

print(f"Wrote formatted notes from {len(previous_days)} non-empty days in {month_name} to {output_file}")