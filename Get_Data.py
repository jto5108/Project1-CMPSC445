import csv
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------
# SETUP SELENIUM
# ----------------------------
options = Options()
options.add_argument("--headless=new")   # headless mode
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
input_file = "ThumbnailScrape.csv"
output_file = "Youtube_Data.csv"
start_row = 1  # change if you want to skip rows

# ----------------------------
# CATEGORY KEYWORDS
# ----------------------------
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

# ----------------------------
# FUNCTION TO ASSIGN CATEGORY
# ----------------------------
def assign_category(title, tag):
    text = f"{title} {tag}".lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return category
    return "General Gaming"

# ----------------------------
# READ INPUT AND WRITE OUTPUT
# ----------------------------
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:  # always overwrite

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["Comments", "FirstTag", "Category"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()  # write header once

    total_rows = sum(1 for _ in open(input_file, encoding="utf-8"))
    infile.seek(0)  # reset reader after counting

    for i, row in enumerate(reader, start=1):
        if i < start_row:
            continue

        url = row.get("URL", "")
        if not url:
            continue  # skip rows without URL

        try:
            driver.get(url)
            time.sleep(2)  # let page load
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(1)

            # comment count
            try:
                comment_count = driver.find_element(By.XPATH, '//h2[@id="count"]/yt-formatted-string').text
            except:
                comment_count = "N/A"

            # first tag
            try:
                first_tag = driver.find_element(By.XPATH, '//meta[@property="og:video:tag"]').get_attribute("content")
            except:
                first_tag = "N/A"

        except:
            comment_count = "N/A"
            first_tag = "N/A"

        # assign category
        category = assign_category(row.get("Title", ""), first_tag if first_tag != "N/A" else "")

        # write row to CSV
        row["Comments"] = comment_count
        row["FirstTag"] = first_tag
        row["Category"] = category
        writer.writerow(row)

        # print progress
        print(f"[{i}/{total_rows}] URL: {url} | Comments: {comment_count} | FirstTag: {first_tag} | Category: {category}")

driver.quit()
print("Scraping complete. Data saved to", output_file)
