import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os

# ----------------------------
# SETUP SELENIUM (headless + container-friendly)
# ----------------------------
options = Options()
options.add_argument("--headless=new")  # headless mode
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
# LOAD EXISTING URLS (avoid duplicates across runs)
# ----------------------------
video_urls = set()
if os.path.exists(csv_file):
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) > 1:
                video_urls.add(row[1])

# ----------------------------
# PREPARE CSV
# ----------------------------
file_exists = os.path.exists(csv_file)
csv_f = open(csv_file, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_f)
if not file_exists:
    writer.writerow(["Title", "URL", "Channel", "Views", "Upload Date"])

# ----------------------------
# SCRAPE LOOP
# ----------------------------
for topic in queries:
    driver.get(f"https://www.youtube.com/results?search_query={topic}&sp=EgIQAQ%253D%253D")
    
    # wait for videos to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//a[@id="video-title"]'))
        )
    except:
        continue

    prev_count = -1
    while len(video_urls) < max_videos and len(video_urls) != prev_count:
        prev_count = len(video_urls)
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)  # small wait for new content

        videos = driver.find_elements(By.XPATH, '//a[@id="video-title"]')

        for v in videos:
            title = v.get_attribute("title")
            url = v.get_attribute("href")

            if not url or "/shorts/" in url or url in video_urls:
                continue  # skip shorts & duplicates

            video_urls.add(url)

            # container traversal
            try:
                container = v.find_element(By.XPATH, "./../../..")
            except:
                container = None

            # extract details
            try:
                channel = container.find_element(By.XPATH, './/*[@id="channel-name"]').text if container else "N/A"
            except:
                channel = "N/A"
            try:
                views = container.find_element(By.XPATH, './/span[contains(text(), "views")]').text if container else "N/A"
            except:
                views = "N/A"
            try:
                upload_date = container.find_element(By.XPATH, './/span[contains(text(), "ago")]').text if container else "N/A"
            except:
                upload_date = "N/A"

            writer.writerow([title, url, channel, views, upload_date])

        if len(video_urls) >= max_videos:
            break

# ----------------------------
# CLEANUP
# ----------------------------
csv_f.close()
driver.quit()
