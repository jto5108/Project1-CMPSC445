import csv
import time
import requests
from bs4 import BeautifulSoup

# ----------------------------
# FILE SETUP
# ----------------------------
input_file = "ThumbnailScrape.csv"
output_file = "Youtube_Data.csv"
start_row = 897  # continue from this row

# User-Agent header to mimic a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "a", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["Comments", "FirstTag"]  # add new columns
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    # Write header only if file is empty
    if outfile.tell() == 0:
        writer.writeheader()

    for i, row in enumerate(reader, start=1):
        if i < start_row:
            continue

        url = row.get("URL")
        print(f"[{i}] Scraping {url}")

        comments = "N/A"
        first_tag = "N/A"

        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                print(f"Failed to fetch page: {response.status_code}")
            else:
                soup = BeautifulSoup(response.text, "html.parser")

                # ----------------------------
                # COMMENTS COUNT
                # ----------------------------
                comment_elem = soup.select_one('h2#count yt-formatted-string')
                if comment_elem:
                    comments = comment_elem.get_text(strip=True)

                # ----------------------------
                # FIRST TAG
                # ----------------------------
                tag_elem = soup.select_one('meta[property="og:video:tag"]')
                if tag_elem:
                    first_tag = tag_elem.get("content", "N/A")

        except Exception as e:
            print(f"Error scraping {url}: {e}")

        # Save enriched row
        row["Comments"] = comments
        row["FirstTag"] = first_tag
        writer.writerow(row)
        print(f"Done {i}")

        time.sleep(1)  # be polite, avoid rate-limiting

print("Scraping complete. Data saved to", output_file)
