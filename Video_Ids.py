from googleapiclient.discovery import build
import pandas as pd
import os

# ---- CHANGE 1: Use your API key directly instead of os.getenv ----
api = build("youtube", "v3", developerKey="AIzaSyDejAMtAIGAo4GLDNBar764UQ0Ty6Euago")  # <-- hardcoded your key

videos = []
next_page = None

while True:
    res = api.search().list(
        part="id",
        q="Marvel Rivals Overwatch",
        type="video",
        maxResults=50,  # 50 is the maximum per request
        pageToken=next_page
    ).execute()

    for item in res["items"]:
        videos.append(item["id"]["videoId"])

    next_page = res.get("nextPageToken")
    if not next_page or len(videos) >= 3000:  # stop at 3000 videos
        break

print(videos)
print(len(videos))

df = pd.DataFrame(videos, columns=["videoId"])  # ---- CHANGE 2: add column name for clarity ----
csv_path = "Video_Ids.csv"

# ---- CHANGE 3: Write CSV with header only if file does not exist ----
if not os.path.isfile(csv_path):
    df.to_csv(csv_path, index=False)  # first time, write headers
else:
    df.to_csv(csv_path, mode='a', header=False, index=False)  # append without headers
