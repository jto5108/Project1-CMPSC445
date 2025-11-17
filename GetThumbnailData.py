import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os

# ----------------------------
# SETUP SELENIUM (headless + fast)
# ----------------------------
options = Options()
# options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ----------------------------
# SEARCH QUERIES
# ----------------------------
queries = [
    "Gaming",
    "Xbox Games",
    "PlayStation Games",
    "Nintendo Games",
    "Upcoming Games 2026",
    "Best Games of 21st century",
    "Minecraft",
    "VR games",
    "Best arcade games",
    "Best phone games",
    "Call of Duty",
    "Rainbow 6 siege",
    "Rocket League",
    "Overwatch",
]


max_videos = 4000
csv_file = "ThumbnailScrape.csv"

# ----------------------------
# PREPARE CSV
# ----------------------------
file_exists = os.path.exists(csv_file)
with open(csv_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Title", "URL", "Channel", "Views", "Upload Date"])

# ----------------------------
# SCRAPE LOOP
# ----------------------------
video_urls = set()  # track unique URLs

for topic in queries:
    print(f"\n🔍 Searching for: {topic}")
    driver.get(f"https://www.youtube.com/results?search_query={topic}&sp=EgIQAQ%253D%253D")
    time.sleep(3)

    prev_count = -1
    while len(video_urls) < max_videos and len(video_urls) != prev_count:
        prev_count = len(video_urls)

        # scroll down
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)

        videos = driver.find_elements(By.XPATH, '//a[@id="video-title"]')

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for v in videos:
                title = v.get_attribute("title")
                url = v.get_attribute("href")

                if not url or "/shorts/" in url or url in video_urls:
                    continue  # skip shorts & duplicates

                video_urls.add(url)

                container = v.find_element(By.XPATH, "./../../..")
                try:
                    channel = container.find_element(By.XPATH, './/*[@id="channel-name"]').text
                except:
                    channel = "N/A"
                try:
                    views = container.find_element(By.XPATH, './/span[contains(text(), "views")]').text
                except:
                    views = "N/A"
                try:
                    upload_date = container.find_element(By.XPATH, './/span[contains(text(), "ago")]').text
                except:
                    upload_date = "N/A"

                writer.writerow([title, url, channel, views, upload_date])
                print(f"Saved: {title} | {url}")

        print(f"Collected so far: {len(video_urls)}")

        if len(video_urls) >= max_videos:
            break

driver.quit()
print("\nDone. Check ThumbnailScrape.csv for results.")
