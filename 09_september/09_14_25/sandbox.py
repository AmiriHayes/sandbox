# try scraping professor homepages from the HTML of a CMU directory
# because looking them all up one by one is annoying (used SerpAPI)

import re
import os
import time
import random
import json
from datetime import datetime
import requests
import pandas as pd

professors = """
<div class="form_responses"><select size="1" id="form_77867799-01e1-4bf7-99eb-d23845090c92" name="form_77867799-01e1-4bf7-99eb-d23845090c92" autocomplete="off"><option></option><option value="fb32f7ab-ee34-4fc6-aaee-f74849cce526" data-text="Aaditya Ramdas">Aaditya Ramdas</option><option value="c63572b3-2fdf-46ee-b834-e03df26d11f5" data-text="Aarti Singh">Aarti Singh</option><option value="110b71c0-dcda-47e4-a890-4cb66025b723" data-text="Aditi Raghunathan">Aditi Raghunathan</option><option value="dda8da44-f7fb-49e6-bad9-b16c293b6965" data-text="Alan Montgomery">Alan Montgomery</option><option value="c6de10ad-40de-46fb-ad77-e2d3f69ec4de" data-text="Albert Gu">Albert Gu</option><option value="41621c6a-04f2-4786-a405-0f9b30b0ce0f" data-text="Ameet Talwalkar">Ameet Talwalkar</option><option value="0077faf2-1bd5-4e53-aef0-f19119cfa2fd" data-text="Andrej Risteski">Andrej Risteski</option><option value="fce51c5e-fa89-4524-9cc9-aab2b7edc2c3" data-text="Andrew Ilyas">Andrew Ilyas</option><option value="2d11fbb2-9bb1-40cb-a440-b572395e17d9" data-text="Ann Lee">Ann Lee</option><option value="74fe451f-4004-4ff3-b50c-b9957e9fd8cb" data-text="Aran Nayebi">Aran Nayebi</option><option value="d2a5a1ec-191d-4a64-a2fe-1c3117ec3408" data-text="Artur Dubrawski">Artur Dubrawski</option><option value="142ae252-42de-432b-89f1-b2a6203ffb64" data-text="Aviral Kumar">Aviral Kumar</option><option value="e958b1ed-ead9-4118-b354-d937ca506e81" data-text="Barnabás Poczos">Barnabás Poczos</option><option value="dd16fac7-6c68-400e-a084-d85f2143d45a" data-text="Beidi Chen">Beidi Chen</option><option value="210487f2-fb6e-4220-94f6-6bd1e476d0bc" data-text="Bhiksha Raj">Bhiksha Raj</option><option value="70302e33-4e0b-42f9-8b5f-4a5e333e4393" data-text="Bryan Wilder">Bryan Wilder</option><option value="fea4c57f-d449-48df-862c-3779fd488ea2" data-text="Chenyan Xiong">Chenyan Xiong</option><option value="1f490a06-712d-4df6-b478-dde4c7b3a6c7" data-text="Christos Faloutsos">Christos Faloutsos</option><option value="034d3a74-c81d-4be5-a946-3b011d2e5b7d" data-text="Daphne Ippolito">Daphne Ippolito</option><option value="502b32de-3c65-4018-8e3d-cf4024ec3aca" data-text="David Held">David Held</option><option value="58343e0b-7ad2-4af4-8e69-30f13bc5af67" data-text="Deepak Pathak">Deepak Pathak</option><option value="849d07f9-f006-43a5-9f8e-1b8260c5f2e5" data-text="Drew Bagnell">Drew Bagnell</option><option value="86c47c21-3675-494a-a282-238499da2273" data-text="Eric Xing">Eric Xing</option><option value="888993cd-7014-41db-b7eb-995efc2e4379" data-text="Fei Fang">Fei Fang</option><option value="4d97e463-fc90-4efc-baae-6dcc6b72eb5f" data-text="Gauri Joshi">Gauri Joshi</option><option value="0d548e21-7508-49d4-9fe0-468e14a22a3d" data-text="Geoff Gordon">Geoff Gordon</option><option value="48dbc1c3-cf9d-4626-9c0e-32a12fdd2678" data-text="Giulia Fanti">Giulia Fanti</option><option value="499851d7-5cbd-4dba-b705-c1ce150c898a" data-text="Graham Neubig">Graham Neubig</option><option value="58b19a02-fc81-4493-a99c-ee83ff6b96b1" data-text="Hao Liu">Hao Liu</option><option value="cbfbed20-e8de-4681-8ce1-a4810b89ff8e" data-text="Henry Chai">Henry Chai</option><option value="268f8b5e-bd9f-40ff-aad7-c460e54a2793" data-text="Hoda Heidari">Hoda Heidari</option><option value="bd0d9be0-391c-4c70-a7ed-3c363581b872" data-text="Jeff Schneider">Jeff Schneider</option><option value="af10fd71-46c9-426e-a156-67f706be4d5c" data-text="Jian Ma">Jian Ma</option><option value="4d8fcc0d-5579-4ce5-84eb-ee1f4a3ed941" data-text="Jun-Yan Zhu">Jun-Yan Zhu</option><option value="e9b1ed48-0d05-4f2a-a466-4184cc7c1246" data-text="Katerina Fragkiadaki">Katerina Fragkiadaki</option><option value="c32a738e-23e5-45dc-8e3f-1e9f42f748ad" data-text="Katia Sycara">Katia Sycara</option><option value="5318bbb1-92f0-43c3-bec7-e96dc8df3972" data-text="Kun Zhang">Kun Zhang</option><option value="0d4c9491-87b8-48bf-ace5-2af2bfbd8c23" data-text="L.P. Morency">L.P. Morency</option><option value="6055e8c8-2fdd-4e75-b185-6feaf6adcc89" data-text="Larry Wasserman">Larry Wasserman</option><option value="94b49d21-ceee-4784-8484-a07277470b9b" data-text="Leila Wehbe">Leila Wehbe</option><option value="5a7a42ca-41a3-4950-8ed1-1cea5d86dc52" data-text="Leman Akoglu">Leman Akoglu</option><option value="af5f90cb-83e1-4db5-b714-5ee77d8306e2" data-text="Matt Gormley">Matt Gormley</option><option value="68db34d6-1acb-4007-9254-2c0845c6c9b6" data-text="Max Simchowitz">Max Simchowitz</option><option value="dde6c605-838e-447c-b2f5-245160f73ab6" data-text="Nicholas Boffi">Nicholas Boffi</option><option value="06646325-5fbe-4e03-bd43-3f35dbffdf21" data-text="Nihar Shah">Nihar Shah</option><option value="886d434b-bb5e-4dd9-b4af-f76e80d92acb" data-text="Nina Balcan">Nina Balcan</option><option value="fdaf0179-5fe4-4b17-a22a-1ddd9d097252" data-text="Pat Virtue">Pat Virtue</option><option value="673323c4-57c5-4a57-9714-8767ededd21c" data-text="Pradeep Ravikumar">Pradeep Ravikumar</option><option value="ec2b2e1a-4302-4117-8f3d-f915b5efc29a" data-text="Rayid Ghani">Rayid Ghani</option><option value="c8df9bc3-fc07-4e03-8657-0a53e134eb8c" data-text="Robert Kass">Robert Kass</option><option value="ae7cf532-42ae-4e36-a1e0-8313c900c90b" data-text="Roni Rosenfeld">Roni Rosenfeld</option><option value="f931cdf7-5e7a-42d5-9382-cb0b58e1386b" data-text="Ruslan Salakhutdinov">Ruslan Salakhutdinov</option><option value="cf35adab-242d-41ec-9268-313fced89b28" data-text="Sivaraman Balakrishnan">Sivaraman Balakrishnan</option><option value="7b82d5ce-d59b-4e90-af59-d949f2ae3982" data-text="Steven Wu">Steven Wu</option><option value="704cdacd-375a-4afa-8102-99f14bd48c0f" data-text="Tianqi Chen">Tianqi Chen</option><option value="06730e7d-bce4-47dc-9db3-62de863fc411" data-text="Tim Dettmers">Tim Dettmers</option><option value="0eee7766-6dc3-4e31-be70-a41c8e27e6d8" data-text="Tom Mitchell">Tom Mitchell</option><option value="4ddc7efe-7f15-4a8f-851a-45bde467fe71" data-text="Tuomas Sandholm">Tuomas Sandholm</option><option value="5e64aeda-1a47-45d4-b8c7-67613abff7a3" data-text="Vincent Conitzer">Vincent Conitzer</option><option value="886f7734-0a44-42d5-9492-66789c8e8007" data-text="Virginia Smith">Virginia Smith</option><option value="41422d82-f508-48dc-9d9e-696e24969ab0" data-text="William Cohen">William Cohen</option><option value="40ac0068-3bd7-4fc9-9669-d1d2dff8817a" data-text="Yiming Yang">Yiming Yang</option><option value="a97da1b6-201a-4117-87c1-d8f4e4694b25" data-text="Yuejie Chi">Yuejie Chi</option><option value="83d85268-8cf6-4fa0-8de4-1070c8e13ceb" data-text="Zachary Lipton">Zachary Lipton</option><option value="ba98e2a3-bd8a-4c72-9d4e-e2d3cf320b66" data-text="Zico Kolter">Zico Kolter</option></select></div>
"""
professor_names = re.findall(r'data-text="([^"]+)"', professors)
print(professor_names, "\n")

API_KEY = "07785250d5b46ce6370e107158b5bfbc797c7a9dfc651637682980c1c6b75413"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
session = requests.Session()

def search_homepage(name, max_retries=3):
    """
    Search SerpApi for `name site:cmu.edu` and return the first result URL,
    or None if nothing usable is found.
    """
    query = f"{name} site:cmu.edu"
    params = {
        "q": query,
        "engine": "google",   # SerpApi engine; google tends to be reliable here
        "api_key": API_KEY,
    }

    for attempt in range(max_retries):
        try:
            resp = session.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"[WARN] network error for '{name}': {e}. retrying in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"[WARN] 429 rate-limited for '{name}'. backing off {wait}s.")
            time.sleep(wait)
            continue
        try:
            data = resp.json()
        except ValueError:
            print(f"[WARN] Non-JSON response for '{name}' (status {resp.status_code}).")
            return None

        # Try to extract first organic result link (common for SerpApi Google responses)
        organic = data.get("organic_results") or []
        if organic and isinstance(organic, list):
            first = organic[0]
            link = first.get("link") or first.get("url")
            if link:
                return link

        # If no organic_results found, check other useful places
        # (answer_box, top_results, knowledge_graph, etc.)
        # Keep checks minimal but flexible:
        # - answer_box (sometimes has 'link')
        ab = data.get("answer_box") or {}
        if isinstance(ab, dict):
            link = ab.get("link") or ab.get("source") or None
            if link:
                return link

        # top_results may exist
        top = data.get("top_results") or []
        if top and isinstance(top, list):
            first = top[0]
            link = first.get("link") or first.get("url")
            if link:
                return link

        # No usable URL found in this response
        return None

    # if retries exhausted
    return None


# Output dict: name -> dict (you asked for dictionaries as values)
homepages = {}

# ensure homepages.json exists or create on first write
JSON_PATH = "homepages.json"

for name in professor_names:
    if name in homepages: continue

    url = search_homepage(name)
    # break immediately if requested and url is None
    if url is None:
        print(f"[STOP] No URL found for '{name}'. Stopping as requested.")
        break

    # store as a dictionary value
    homepages[name] = {"url": url}
    print(f"{name}: {homepages[name]}")

    # record into pandas and append/save to homepages.json
    row = {
        "name": name,
        "url": url,
        "retrieved_at": datetime.utcnow().isoformat() + "Z"
    }
    row_df = pd.DataFrame([row])

    if os.path.exists(JSON_PATH):
        try:
            existing = pd.read_json(JSON_PATH, orient="records")
            combined = pd.concat([existing, row_df], ignore_index=True)
        except ValueError:
            # file exists but invalid/empty — overwrite with current row only
            combined = row_df
    else:
        combined = row_df

    # Write back full JSON (records orientation)
    combined.to_json(JSON_PATH, orient="records", indent=2)

    # be polite with a small randomized delay
    time.sleep(random.uniform(1.5, 3.5))

# done; print final dictionary
print("\nFinal homepages dictionary (partial if loop was stopped):")
print(json.dumps(homepages, indent=2))
