# scrape yaml files to create summaries for last couple days pushes for Readme.md

import datetime
import yaml
import json
from pathlib import Path

today = datetime.date.today()
month_name = today.strftime("%B").lower()
base_dir = Path("..").resolve()
month_dir = base_dir / month_name
print(f"base dir: {base_dir.cwd()}")
print(f"month dir: {month_dir.cwd()}")

day_folders = []
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
    notes_content += str(json.dumps(data))
    notes_content += "\n"

project_root = Path(__file__).parent.parent.resolve()
output_file = project_root / "docs/calendar.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(notes_content)

print(f"Wrote formatted notes from {len(previous_days)} non-empty days in {month_name} to {output_file}")