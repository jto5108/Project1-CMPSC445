from googleapiclient.discovery import build
import pandas as pd
import os

youtube_api_key = os.getenv("API_KEY")
api = build("youtube", "v3", developerKey=youtube_api_key)

videos = []
next_page = None

while True:
    res = api.search().list(
        part="id",
        q="Marvel Rivals Overwatch",
        type="video",
        maxResults=50,  # 50 is the set max
        pageToken=next_page
    ).execute()

    for item in res["items"]:
        videos.append(item["id"]["videoId"])

    next_page = res.get("nextPageToken")
    if not next_page or len(videos) >= 3000:
        break

print(videos)
print(len(videos))

df = pd.DataFrame(videos)
csv_path = "/Users/jto5108/Project1-CMPSC445/Video_Ids.csv"
df.to_csv(csv_path, mode='a', header=False, index=False)
