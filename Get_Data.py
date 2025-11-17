import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------
# SETUP SELENIUM
# ----------------------------
options = Options()
options.add_argument("--headless=new")   # run headless
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


input_file = "ThumbnailScrape.csv"
output_file = "YT_Video_Data.csv"
start_row = 897
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "a", newline="", encoding="utf-8") as outfile: #KEEP "a" to add to file. "W" will rewrite over

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, row in enumerate(reader, start=1):
        if i < start_row:
            continue

        url = row["URL"]
        print(f"[{i}] Scraping {row}")

        try:
            driver.get(url)
            time.sleep(3)  # let page load

            #Scroll down for comment count
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)

            # ----------------------------
            # COMMENT COUNT
            # ----------------------------
            try:
                comment_count = driver.find_element(By.XPATH, '//h2[@id="count"]/yt-formatted-string').text
            except:
                comment_count = "N/A"

            try:
                first_tag = driver.find_element(By.XPATH, '//meta[@property="og:video:tag"]').get_attribute("content")
            except:
                first_tag = "N/A"

        except Exception as e:
            print(f"Error on {url}: {e}")
            comment_count = "N/A"
            first_tag = "N/A"

        # Save enriched row
        row["Comments"] = comment_count
        row["FirstTag"] = first_tag
        writer.writerow(row)
        print(f"Done: {i/3292}")

print("Step 2 complete. Data saved to", output_file)
driver.quit()
