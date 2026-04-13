import time
import uuid
import re
import os
import json
import base64
import random
import requests
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# CONFIG
# =========================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17 Safari/605.1.15",
]

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

if not os.path.exists("images"):
    os.makedirs("images")

# =========================
# DRIVER ESTABLE
# =========================
def create_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options

    options = Options()

    # 🔥 MODO DEBUG (IMPORTANTE)
    # Primero prueba SIN headless
    options.add_argument("--start-maximized")

    # ✅ estabilidad total
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")

    # ✅ evita crash silencioso
    options.add_argument("--remote-debugging-port=9222")

    # ⚠️ MUY IMPORTANTE
    options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    service = Service(r"C:\chromedriver\chromedriver.exe")

    driver = webdriver.Chrome(service=service, options=options)

    driver.set_page_load_timeout(30)

    return driver
# =========================
# SCRAPER HELPERS
# =========================
def get_price(driver):
    try:
        el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//span[contains(@class,'price')]"))
        )
        return float(re.sub(r"[^\d]", "", el.text))
    except:
        return 0

def get_name(driver):
    try:
        return driver.find_element(By.TAG_NAME, "h1").text
    except:
        return ""

def get_brand(driver):
    try:
        return driver.find_element(By.XPATH, "//a[contains(@href,'/shopping/')]").text
    except:
        return ""

def download_image(url, path):
    try:
        r = requests.get(url, headers=get_headers(), timeout=10)
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except:
        return False

def get_images(driver, product_id):
    imgs = driver.find_elements(By.XPATH, "//img")
    paths = []
    used = set()

    for i, img in enumerate(imgs):
        src = img.get_attribute("src") or ""
        if "jpg" in src and src not in used:
            used.add(src)
            path = f"images/{product_id}_{i}.jpg"
            if download_image(src, path):
                paths.append("/" + path)
        if len(paths) >= 4:
            break

    return paths

# =========================
# OPENAI
# =========================
def analyze_images(image_paths, name, brand):
    if not OPENAI_API_KEY or not image_paths:
        return {}

    imgs = []
    for path in image_paths[:2]:
        with open(path.lstrip("/"), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        imgs.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    prompt = f"""
Producto: {name}
Marca: {brand}

Devuelve JSON con:
description, productType, colors
"""

    try:
        res = openai_client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": imgs + [{"type": "text", "text": prompt}]
            }]
        )
        raw = res.output[0].content[0].text
        return json.loads(raw)
    except:
        return {}

# =========================
# SCRAPE
# =========================
def scrape(url):
    driver = None
    try:
        driver = create_driver()
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        name = get_name(driver)
        brand = get_brand(driver)
        price = get_price(driver)

        product_id = str(uuid.uuid4())[:8]
        images = get_images(driver, product_id)

        ai = analyze_images(images, name, brand)

        return {
            "id": product_id,
            "name": name,
            "brand": brand,
            "price": price,
            "images": "|".join(images),
            "description": ai.get("description", ""),
            "productType": ai.get("productType", ""),
            "colors": ai.get("colors", "")
        }

    except Exception as e:
        return {"error": str(e), "url": url}

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# =========================
# EXCEL SEGURO
# =========================
def save_excel(data):
    folder = "exports"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file = os.path.join(folder, f"products_{int(time.time())}.xlsx")

    df = pd.DataFrame(data)
    df.to_excel(file, index=False)

    return file

# =========================
# REQUEST
# =========================
class ScrapeRequest(BaseModel):
    urls: list[str]

# =========================
# ENDPOINT
# =========================
@app.post("/scrape")
def scrape_api(req: ScrapeRequest):
    results = []

    for url in req.urls:
        results.append(scrape(url))

    file = save_excel(results)

    return {
        "status": "ok",
        "products": len(results),
        "file": file,
        "data": results
    }