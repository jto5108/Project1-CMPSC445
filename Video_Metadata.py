from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os, csv, datetime, isodate, time

key = os.getenv("API_KEY")
api = build("youtube", "v3", developerKey=key)

video_ids = []
csv_path = "/Users/andrewherman/CMP 455/Project 1/API_Code/Video_Ids.csv"
with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_ids.append(row["VideoId"])

data = []
channel_cache = {}
for i in range(0, len(video_ids), 50):  # batch 50 per request
    batch = video_ids[i:i + 50]
    try:
        res = api.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch)
        ).execute()
    except HttpError as e:
        print("HTTP Error:", e)
        time.sleep(10)
        continue

    for item in res["items"]:
        snippet = item["snippet"]
        stats = item["statistics"]
        details = item["contentDetails"]

        # Duration in seconds
        duration_sec = int(isodate.parse_duration(details["duration"]).total_seconds())

        # Days since upload
        published = datetime.datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00"))
        days_since = (datetime.datetime.now(datetime.timezone.utc) - published).days

        # Channel subscriber count
        ch_id = snippet["channelId"]
        if ch_id not in channel_cache:
            ch_res = api.channels().list(part="statistics", id=ch_id).execute()
            subs = ch_res["items"][0]["statistics"].get("subscriberCount")
            channel_cache[ch_id] = subs
        else:
            subs = channel_cache[ch_id]

        data.append({
            "video_id": item["id"],
            "title": snippet["title"],
            "duration_seconds": duration_sec,
            "days_since_upload": days_since,
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
            "comments": stats.get("commentCount"),
            "channel_title": snippet["channelTitle"],
            "subscribers": subs
        })

csv_path = "/Users/andrewherman/CMP 455/Project 1/API_Code/youtube_videos.csv"
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    if f.tell() == 0:
        writer.writeheader()
    writer.writerows(data)

print(f"Saved {len(data)} videos.")
