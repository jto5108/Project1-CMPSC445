import pandas as pd

#Get csv data
df = pd.read_csv("Youtube_Video_Data.csv")

"""
Code to add category column
Only needed once
"""
# df["Category"] = "Unclassed"
# df.to_csv("Youtube_Video_Data.csv", index=False)

categories = {
    "Xbox": ["xbox", "game pass", "series x", "series s", "microsoft"],
    "PlayStation": ["playstation", "ps5", "ps4", "ps1", "ps2", "ps3", "play station", "sony", "spiderman"],
    "Nintendo": ["nintendo", "switch", "zelda", "mario", "mario party", "luigi"],
    "VR": ["vr", "oculus", "meta quest", "virtual reality"],
    "Arcade": ["arcade", "retro", "pinball"],
    "Mobile": ["mobile", "phone game", "ios", "android"],
    "Minecraft": ["minecraft", "steve", "creeper"],
    "CallOfDuty": ["call of duty", "cod", "modern warfare", "black ops", "bo4", "bo6", "bo7"],
    "RainbowSix": ["rainbow six", "siege", "r6"],
    "RocketLeague": ["rocket league", "rlcs", "musty"],
    "Overwatch": ["overwatch", "ow2"],
}

#Define class by using keywords from title and tags found in video
def assign_category(video_title, tag):
    video_title = f"{video_title} {tag}".lower()
    for category in categories:
        for kw in categories[category]:
            if kw in video_title:
                return category
    return "General Gaming"

#Call function to assign category to row
df["Category"] = df.apply(lambda row: assign_category(row["Title"], row.get("FirstTag", "")), axis=1)

#Write back to csv
df.to_csv("Youtube_Video_Data.csv", index=False)
