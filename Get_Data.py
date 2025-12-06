import csv
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------
# SETUP SELENIUM
# ----------------------------
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 5)  # wait max 5 seconds for elements

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
input_file = "ThumbnailScrape.csv"
output_file = "Youtube_Data.csv"
start_row = 1

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
# CHECK IF CSV EXISTS
# ----------------------------
file_exists = os.path.exists(output_file)

# ----------------------------
# READ INPUT AND WRITE OUTPUT
# ----------------------------
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "a", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["Comments", "FirstTag", "Category"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    for i, row in enumerate(reader, start=1):
        if i < start_row:
            continue

        url = row.get("URL", "")
        if not url:
            continue

        # ----------------------------
        # SCRAPE YOUTUBE
        # ----------------------------
        try:
            driver.get(url)

            # comment count
            try:
                comment_count = wait.until(
                    EC.presence_of_element_located((By.XPATH, '//h2[@id="count"]/yt-formatted-string'))
                ).text
            except TimeoutException:
                comment_count = "N/A"

            # first tag
            try:
                first_tag = wait.until(
                    EC.presence_of_element_located((By.XPATH, '//meta[@property="og:video:tag"]'))
                ).get_attribute("content")
            except TimeoutException:
                first_tag = "N/A"

        except:
            comment_count = "N/A"
            first_tag = "N/A"

        # ----------------------------
        # ASSIGN CATEGORY
        # ----------------------------
        category = assign_category(row.get("Title", ""), first_tag if first_tag != "N/A" else "")

        # ----------------------------
        # WRITE ROW TO CSV
        # ----------------------------
        row["Comments"] = comment_count
        row["FirstTag"] = first_tag
        row["Category"] = category
        writer.writerow(row)

driver.quit()
