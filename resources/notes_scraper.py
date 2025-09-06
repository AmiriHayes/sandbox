# scrape yaml files to create summaries for last couple days pushes for Readme.md

import datetime
import yaml
from pathlib import Path

today = datetime.date.today()
month_name = today.strftime("%B").lower()
base_dir = Path("/home/runner/work/sandbox/sandbox").resolve()

august = Path("/home/runner/work/sandbox/sandbox/08_august")
september = Path("/home/runner/work/sandbox/sandbox/09_september")
months = [august, september]

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

day_folders.sort(key=lambda x: x[0])
last_five = [f for d, f in reversed(day_folders) if d <= today][:5]

notes_content = ""
notes_content += "# Amiri's Sandbox Repository\n\n"
notes_content += "**Goal:** Make a folder every day and write ~100 lines of code in it to practice a skill or learn more about a topic in a manageable way. Topics include ML, data engineering, cloud tech, full stack devlopment and so on. <br> \n"
notes_content += "\n **Repo Docs Site:** [Docs History](https://amirihayes.github.io/sandbox/) <br><br> \n\n"
notes_content += "#### Automatic Updates | Last 5 Days: \n\n"

for folder in last_five:
    notes_file = folder / "notes.yaml"
    with open(notes_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    month, day, year = folder.name.split("_")
    formatted_date = f"{month} / {day} / {year}"
    topic = data.get("Topic", "N/A")
    worked_on = data.get("Goal", "N/A")
    read_up_on = data.get("Todays_Paper", "N/A")
    link = data.get("Deliverable_Link", "N/A")

    notes_content += f"<em>{formatted_date}: </em>  \n"
    notes_content += f"---  Topic: {topic}  \n"
    notes_content += f"---  Worked on: {worked_on}  \n"
    if link:
        notes_content += f"---  Read up on: {read_up_on}  \n"
        notes_content += f"---  Link: {link}  \n\n"
    else:
        notes_content += f"---  Read up on: {read_up_on}  \n\n"

output_file = Path("/home/runner/work/sandbox/sandbox/Readme.md")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(notes_content)

print(f"Wrote formatted notes from {len(last_five)} non-empty days in {month_name} to {output_file}")
