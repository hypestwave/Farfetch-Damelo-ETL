import uuid
import re
import os
import json
import base64
import random
import requests
import pandas as pd
import cv2
import numpy as np
import onnxruntime as ort
import time
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
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
# CACHE
# =========================
def load_cache():
    if os.path.exists("cache.json"):
        with open("cache.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open("cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
def scrape_with_retry(url, retries=3):
    for attempt in range(retries):
        try:
            result = scrape(url)

            if "error" not in result:
                return result

        except Exception as e:
            print(f"⚠️ Retry {attempt+1} error:", e)

        time.sleep(random.uniform(2, 5))

    return {
        "error": "max retries reached",
        "url": url
    }

lock = threading.Lock()

def process_url(url, cache):
    if url in cache:
        print(f"⚡ Cache hit: {url}")
        return cache[url]

    print(f"🔎 Scraping: {url}")

    result = scrape_with_retry(url)

    with lock:
        cache[url] = result
        save_cache(cache)

    return result

def run_crawler(max_workers=5):
    urls = list(set(load_urls_from_txt()))
    cache = load_cache()

    print(f"\n🚀 Total URLs: {len(urls)}")
    print(f"⚡ Workers: {max_workers}\n")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_url, url, cache): url
            for url in urls
        }

        for future in as_completed(futures):
            url = futures[future]

            try:
                result = future.result()

                if "error" in result:
                    print(f"❌ {url}")
                else:
                    print(f"✅ {result.get('name')}")

                results.append(result)

                # 🔥 guardado incremental excel
                if len(results) % 3 == 0:
                    save_excel(results)

            except Exception as e:
                print(f"💥 Thread error: {e}")

    # =========================
    # 💾 EXPORT FINAL
    # =========================
    file = save_excel(results)

    print("\n📦 Export final:")
    print(file)

    return results



# =========================
# reimplemetation with direct api
# =========================

MODEL_PATH = "shoe_classifier.onnx"
session = None

def get_onnx_session():
    global session

    if session is None:
        if not os.path.exists(MODEL_PATH):
            return None
        session = ort.InferenceSession(MODEL_PATH)

    return session


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def classify_shoe_onnx(image_path):
    try:
        sess = get_onnx_session()

        if sess is None:
            return "Zapatillas"  # fallback

        input_tensor = preprocess(image_path)

        outputs = sess.run(None, {"input": input_tensor})
        probs = outputs[0][0]

        idx = np.argmax(probs)
        return SHOE_CATEGORIES[idx]

    except:
        return "Zapatillas"

def extract_product_id(url):
    match = re.search(r'item-(\d+)', url)
    return match.group(1) if match else None

def load_urls_from_txt(filepath="products.txt"):
    if not os.path.exists(filepath):
        raise Exception(f"No existe {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    return urls

def run_from_txt():
    urls = load_urls_from_txt()

    print(f"🚀 Procesando {len(urls)} productos...\n")

    results = []

    for i, url in enumerate(urls, 1):
        print(f"\n🔎 [{i}/{len(urls)}] {url}")

        try:
            product = scrape(url)

            # 🔥 validar error
            if "error" in product:
                print(f"❌ Error: {product['error']}")
            else:
                print(f"✅ OK: {product.get('name')}")

            results.append(product)

        except Exception as e:
            print(f"💥 Exception: {e}")
            results.append({
                "error": str(e),
                "url": url
            })

        # 🔥 delay humano (anti-bloqueo)
        time.sleep(random.uniform(2, 4))

    # =========================
    # 💾 EXPORT EXCEL
    # =========================
    file = save_excel(results)

    print("\n📦 Export completo:")
    print(file)

    return results

def fetch_product_api(url):
    product_id = extract_product_id(url)

    api_url = f"https://www.farfetch.com/api/product/{product_id}"

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
        "Referer": url
    }

    r = requests.get(api_url, headers=headers)

    if r.status_code != 200:
        return None

    data = r.json()

    return data

###new function
def parse_product_api(data):
    product = data.get("product", {})
    name = product.get("shortDescription", "")
    brand = product.get("brand", {}).get("name", "")

    base_price = product.get("price", {}).get("value", 0)

    images = [
        img.get("url")
        for img in product.get("images", [])
    ]

    sizes = []
    quantities = []
    prices = []

    for variant in product.get("variants", []):
        size = variant.get("size", {}).get("name")
        stock = variant.get("stock", {}).get("quantity", 0)

        # 🔥 PRECIO POR VARIANTE
        variant_price = variant.get("price", {}).get("value")

        if size:
            sizes.append(size)
            quantities.append(stock if stock else 0)

            if variant_price:
                prices.append(variant_price)

    # =========================
    # 🔥 LÓGICA CLAVE
    # =========================
    final_price = base_price

    if prices:
        max_variant_price = max(prices)

        # 👉 si hay una talla más cara → usar ese precio
        if max_variant_price > base_price:
            final_price = max_variant_price

    return {
        "name": name,
        "brand": brand,
        "price": final_price,
        "images": images,
        "sizes": sizes,
        "quantities": quantities
    }
 
SHOE_CATEGORIES = [
    "Tacones", "Botas", "Mocasines", "Mules",
    "Zapatillas", "Sandalias", "Bailarinas",
    "Chanclas", "Pantuflas",
    
    # 🔥 extras recomendadas
    "Sneakers", "Loafers", "Derby",
    "Oxford", "Slides", "Clogs",
    "Espadrilles", "Ankle Boots",
    "Knee Boots"
]

# =========================
# categories of shoes
# =========================
VALID_SHOE_TYPES = [
    "Tacones",
    "Botas",
    "Mocasines",
    "Mules",
    "Zapatillas",
    "Sandalias",
    "Bailarinas",
    "Chanclas",
    "Pantuflas",

    # 🔥 extras recomendadas
    "Sneakers",
    "Loafers",
    "Derbies",
    "Oxfords",
    "Slides",
    "Espadrilles",
    "Platform",
    "Wedges",
    "Ankle Boots",
    "Knee-high Boots"
]

# =========================
# OPENCV
# =========================
def get_product_type_final(name, details, images):
    # 1. REGLAS y OLLAMA  (rápido y preciso)
    result = get_product_type_hybrid(name, details)
    if result and result != ["Shoes"]:
        return result

    # 3. ONNX (solo si existe)
    USE_ONNX = False
    if USE_ONNX and images:
        vision = detect_product_type_from_image(images)
        if vision:
            return vision

    return ["Zapatillas"]


def classify_with_ollama(name, brand, details, visual_features):
    try:
        prompt = f"""
Eres un clasificador experto en calzado.

Clasifica el producto en UNA categoría válida.

Categorías permitidas:
{", ".join(VALID_SHOE_TYPES)}

Producto: {name}
Marca: {brand}

Características:
{", ".join(details.get("features", []))}

Datos visuales:
- edge_density: {visual_features.get("edge_density")}
- aspect_ratio: {visual_features.get("aspect_ratio")}

Reglas:
- SOLO responde una categoría EXACTA de la lista
- No inventar
- Si dudas → Zapatillas

Respuesta:
"""

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )

        output = res.json().get("response", "").strip()

        # 🔥 normalización
        for cat in VALID_SHOE_TYPES:
            if cat.lower() in output.lower():
                return [cat]

    except:
        pass

    return []

def extract_visual_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {}

        img = cv2.resize(img, (224, 224))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean()

        # proporción
        h, w = gray.shape
        aspect_ratio = w / h

        return {
            "edge_density": float(edge_density),
            "aspect_ratio": float(aspect_ratio)
        }

    except:
        return {}

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    # formato modelo
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)

    return img

def process_product(url):
    data = fetch_product_api(url)

    if not data:
        return {"error": "API failed"}

    parsed = parse_product_api(data)

    product_id = str(uuid.uuid4())[:8]

    # 🔥 descargar imagen principal
    img_url = parsed["images"][0]
    img_path = f"images/{product_id}.jpg"
    download_image(img_url, img_path)

    # 🔥 clasificación IA
    category = classify_shoe_onnx(img_path)

    # 🔥 descripción IA local
    details = {
        "features": [],
        "composition": [],
        "extra": []
    }

    description = generate_description_ollama_pro(
        parsed["name"],
        parsed["brand"],
        details,
        [],
        category
    )

    return {
        "id": product_id,
        "name": parsed["name"],
        "trade": parsed["brand"],
        "price": parsed["price"],
        "images": parsed["images"],
        "size": parsed["sizes"],
        "quantity": parsed["quantities"],
        "productType": [category],
        "description": description,
        "currency": "USD",
        "available": True
    }
# import onnxruntime as ort
# MODEL_PATH = "shoe_classifier.onnx"
# session = ort.InferenceSession(MODEL_PATH)


def get_stealth_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive"
    }

def human_delay():
    time.sleep(random.uniform(2, 5))

PROXIES = [
    "http://user:pass@ip1:port",
    "http://user:pass@ip2:port"
]

def get_proxy():
    return {
        "http": random.choice(PROXIES),
        "https": random.choice(PROXIES)
    }


MODEL_CLASSES = [
    "Zapatillas",
    "Tacones",
    "Botas",
    "Sandalias",
    "Mocasines",
    "Mules",
    "Bailarinas",
    "Chanclas",
    "Pantuflas"
]

VALID_CATEGORIES = [
    "Tacones",
    "Botas",
    "Mocasines",
    "Mules",
    "Zapatillas",
    "Sandalias",
    "Bailarinas",
    "Chanclas",
    "Pantuflas",
    # 🔥 extras recomendados
    "Sneakers",
    "Loafers",
    "Derby",
    "Oxford",
    "Slides",
    "Espadrilles"
]
def get_product_type_hybrid(name, details, images=None):
    import re

    text = f"{name} {' '.join(details.get('features', []))}".lower()
    
    
    
    # =========================
    # 1. REGLAS (FAST PATH)
    # =========================
    rules = {
        "Tacones": ["tacón", "heel", "pump"],
        "Botas": ["bota", "boot"],
        "Mocasines": ["mocasín", "loafer"],
        "Mules": ["mule"],
        "Zapatillas": ["zapatilla", "sneaker", "tenis"],
        "Sandalias": ["sandalia", "sandals"],
        "Bailarinas": ["bailarina", "flat"],
        "Chanclas": ["chancla", "flip flop"],
        "Pantuflas": ["pantufla", "slipper"],
    }

    for category, keywords in rules.items():
        if any(k in text for k in keywords):
            return [category]

    # =========================
    # 2. OLLAMA (IA LOCAL)
    # =========================
    try:
        import requests
        prompt = f"""
Clasifica el tipo de calzado en UNA categoría válida.

Categorías:
Tacones, Botas, Mocasines, Mules, Zapatillas, Sandalias, Bailarinas, Chanclas, Pantuflas, Sneakers, Loafers, Derby, Oxford, Slides, Espadrilles

Producto: {name}
Características: {', '.join(details.get('features', []))}

Responde SOLO con la categoría exacta.
"""

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=5
        )

        if res.status_code == 200:
            response = res.json().get("response", "").strip()

            for cat in VALID_CATEGORIES:
                if cat.lower() in response.lower():
                    return [cat]

    except Exception:
        pass

    # =========================
    # 3. FALLBACK INTELIGENTE
    # =========================
    if "suela" in text:
        return ["Zapatillas"]

    if "piel" in text:
        return ["Mocasines"]

    return ["Shoes"]

def detect_product_type_from_image(image_paths):
    if not image_paths:
        return []

    try:
        img = preprocess_image(image_paths[0].lstrip("/"))
        if img is None:
            return []

        sess = get_onnx_session()
        if not sess:
            return []

        inputs = {sess.get_inputs()[0].name: img}
        outputs = sess.run(None, inputs)

        probs = outputs[0][0]
        idx = int(np.argmax(probs))

        category = MODEL_CLASSES[idx]

        return [category]

    except Exception:
        return []
    
def detect_product_type_fallback(name, details):
    text = f"{name} {' '.join(details.get('features', []))}".lower()

    if "tacón" in text:
        return ["Tacones"]
    if "bota" in text:
        return ["Botas"]
    if "sandalia" in text:
        return ["Sandalias"]
    if "zapat" in text or "sneaker" in text:
        return ["Zapatillas"]
    if "mocas" in text:
        return ["Mocasines"]

    return ["Zapatillas"]

# =========================
# SCRAPER HELPERS
# =========================

def get_price(driver):
    try:
        # ✅ SELECTOR EXACTO FARFETCH
        el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-component='Price']")
            )
        )

        price_text = el.text.strip()

        # 🔥 LIMPIEZA FORMATO LATAM
        # "$ 3.551.309" → "3551309"
        price_clean = (
            price_text
            .replace("$", "")
            .replace("COP", "")
            .replace(".", "")
            .replace(",", "")
            .strip()
        )

        return float(price_clean)

    except Exception:
        pass

    # 🔥 FALLBACK 1 — JSON-LD
    try:
        html = driver.page_source
        match = re.search(r'"price"\s*:\s*"?([0-9]+)"?', html)
        if match:
            return float(match.group(1))
    except:
        pass

    # 🔥 FALLBACK 2 — cualquier texto con $
    try:
        el = driver.find_element(By.XPATH, "//p[contains(text(),'$')]")
        text = el.text
        price_clean = re.sub(r"[^\d]", "", text)
        return float(price_clean)
    except:
        pass

    return 0
  
def get_brand(driver):
    selectors = [
        "//a[@data-ffref='pp_infobrd']",
        "//a[contains(@href,'/shopping/') and contains(@href,'/items.aspx')]"
    ]

    for sel in selectors:
        try:
            el = driver.find_element(By.XPATH, sel)
            if el.text.strip():
                from html import unescape
                return unescape(el.text.strip())
        except:
            continue

    return "Unknown"

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

def get_product_name(driver):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    # 🔥 SELECTOR EXACTO FARFETCH
    try:
        el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-testid='product-short-description']")
            )
        )
        text = el.text.strip()
        if text:
            return text
    except Exception:
        pass

    # 🔥 FALLBACK 1 — h1 (algunos productos lo usan)
    try:
        h1 = driver.find_elements(By.TAG_NAME, "h1")
        for el in h1:
            text = el.text.strip()
            if len(text) > 5:
                return text
    except:
        pass

    # 🔥 FALLBACK 2 — JSON-LD
    try:
        import re, json
        html = driver.page_source
        match = re.search(r'"name"\s*:\s*"([^"]+)"', html)
        if match:
            return match.group(1)
    except:
        pass

    return "Unknown Product"

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

def parse_colors(colors_str):
    result = []

    if not colors_str:
        return result

    parts = colors_str.split("|")

    for p in parts:
        try:
            name, hexcode = p.split(":")
            result.append({
                "name": name.strip(),
                "hex": hexcode.strip()
            })
        except:
            continue

    return result
 
def extract_ref(url):
    match = re.search(r'item-(\d+)', url)
    return match.group(1) if match else ""
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

        name = get_product_name(driver)
        brand = get_brand(driver)
        price = get_price(driver)

        product_id = str(uuid.uuid4())[:8]
        images = get_images(driver, product_id)

        sizes, quantities, size_map = get_sizes_advanced(driver)
        
        accordion_data = get_product_details_from_accordion(driver)
        details = build_details_array(accordion_data)
        ai = analyze_images(images, name, brand)
        colors = parse_colors(ai.get("colors", "")) or []
        product_type = get_product_type_final(
            name,
            accordion_data,
            images
        )
        description = generate_description_ollama_pro(
            name,
            brand,
            accordion_data,
            colors,
            product_type[0] if product_type else "Shoes"
        )
        return {
            "id": product_id,
            "name": name,
            "size": sizes,
            "price": price,
            "quantity": quantities,
            "description": description,
            "image": images[0] if images else "",
            "images": images,
            "colors": colors,
            "productType": product_type,
            "trade": brand,
            "sku": f"{brand}-{product_id}",
            "source": url,
            "externalRef": extract_ref(url),
            "currency": "COP",
            "department": "Women",
            "category": product_type[0] if product_type else "Shoes",
            "details": details,
            "ownerId": "9001",
            "available": True,
            "archived": False,
        }

    except Exception as e:
        return {"error": str(e), "url": url}

    finally:
        if driver:
            driver.quit()
# =========================
# SIZES
# =========================
def get_sizes_advanced(driver):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    import re

    sizes = []
    quantities = []
    prices = []

    try:
        # =========================
        # 1. ABRIR SELECTOR
        # =========================
        selector = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "[data-component='SizeSelectorLabel']")
            )
        )

        driver.execute_script("arguments[0].click();", selector)

        # 🔥 esperar render REAL (no sleep fijo)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//li | //div[@role='option']")
            )
        )

        time.sleep(1)  # micro delay para React hydration

        # =========================
        # 2. CAPTURAR OPCIONES
        # =========================
        options = driver.find_elements(
            By.XPATH,
            "//li | //div[@role='option']"
        )

        for opt in options:
            text = opt.text.strip()

            if not text:
                continue

            # =========================
            # 🔥 EXTRAER SIZE COMPLETA (SIN NORMALIZAR)
            # =========================
            # Detecta formatos:
            # 22.5 Japan (cm)
            # 10 US
            # 44 EU
            size_match = re.search(
                r"(\d+(\.\d+)?\s*(US|EU|UK|JP|CM|KOREA|ITALY|FR|IT|JAPAN|US MEN|US WOMEN|EUROPE|CM))",
                text.upper()
            )

            if not size_match:
                continue

            size = size_match.group(0).strip()

            if size in sizes:
                continue

            sizes.append(size)

            # =========================
            # 🔥 STOCK
            # =========================
            lower = text.lower()

            if "última pieza" in lower:
                quantities.append(1)
            elif "agotado" in lower:
                quantities.append(0)
            else:
                quantities.append(5)

            # =========================
            # 🔥 PRECIO POR VARIANTE
            # =========================
            price_match = re.search(r"\$[\s\d\.,]+", text)

            if price_match:
                price_clean = re.sub(r"[^\d]", "", price_match.group())
                prices.append(int(price_clean))
            else:
                prices.append(0)

        # =========================
        # 3. VALIDACIÓN
        # =========================
        if sizes:
            return sizes, quantities, prices

    except Exception as e:
        print("❌ Error obteniendo tallas:", e)

    # =========================
    # 🔥 FALLBACK API (FIABLE)
    # =========================
    try:
        url = driver.current_url
        data = fetch_product_api(url)

        if data:
            parsed = parse_product_api(data)
            return parsed["sizes"], parsed["quantities"], [parsed["price"]] * len(parsed["sizes"])

    except Exception as e:
        print("❌ API fallback error:", e)

    # =========================
    # 🔥 FALLBACK FINAL
    # =========================
    return [], [], []
# =========================
# DESCRIPTION
# =========================
def get_product_details_from_accordion(driver):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    details = {
        "features": [],
        "composition": [],
        "extra": []
    }

    try:
        # =========================
        # 1. ENCONTRAR SECCIÓN REAL
        # =========================
        section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//section[.//span[contains(text(),'Detalles del producto')]]")
            )
        )

        # =========================
        # 2. TEXTO PRINCIPAL
        # =========================
        paragraphs = section.find_elements(By.XPATH, ".//p")

        for p in paragraphs:
            text = p.text.strip()

            if not text:
                continue

            if "ID" in text:
                details["extra"].append(text)
            else:
                details["features"].append(text)

        # =========================
        # 3. COMPOSICIÓN
        # =========================
        comp = section.find_elements(By.XPATH, ".//h4[contains(text(),'Composición')]/following::p")

        for c in comp:
            text = c.text.strip()
            if text:
                details["composition"].append(text)

        return details

    except Exception as e:
        print("❌ Details error:", e)

    return details


def build_product_context(name, brand, details, colors, category):
    text = f"{name} {' '.join(details.get('features', []))}".lower()

    # 🎯 DETECCIÓN DE SEGMENTO
    segment = "casual"

    if any(k in text for k in ["running", "sport", "trainer", "tenis", "zapat"]):
        segment = "sport"
    elif any(k in text for k in ["cuero", "leather", "italy", "hecho en"]):
        segment = "luxury"

    # 🎨 COLOR
    color = ""
    if colors:
        color = colors[0].get("name", "")

    # 🧵 MATERIAL
    material = ""
    if details.get("composition"):
        material = details["composition"][0]

    return {
        "name": name,
        "brand": brand,
        "category": category,
        "features": details.get("features", [])[:4],
        "material": material,
        "color": color,
        "segment": segment
    }
    
def get_brand_tone(brand):
    brand = brand.lower()

    if any(b in brand for b in ["gucci", "prada", "balenciaga", "valentino"]):
        return "lujo, sofisticado, exclusivo, editorial"
    
    if any(b in brand for b in ["nike", "adidas", "puma", "reebok"]):
        return "dinámico, deportivo, energético"
    
    if any(b in brand for b in ["zara", "h&m", "mango"]):
        return "moderno, accesible, tendencia"
    
    return "premium, elegante, contemporáneo"

def generate_description_ollama_pro(name, brand, details, colors, category):
    import requests
    import re

    ctx = build_product_context(name, brand, details, colors, category)
    tone = get_brand_tone(brand)

    try:
        # =========================
        # 🧩 1. GENERACIÓN BASE
        # =========================
        prompt_base = f"""
Eres copywriter de moda.

Escribe una descripción de producto (150–220 caracteres).

Reglas:
- Español neutro
- No inventar
- No repetir marca más de una vez

Contexto:
Marca: {ctx["brand"]}
Producto: {ctx["name"]}
Categoría: {ctx["category"]}
Segmento: {ctx["segment"]}

Características:
{", ".join(ctx["features"])}

Material:
{ctx["material"]}

Color:
{ctx["color"]}

Salida:
Solo texto.
"""

        res1 = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt_base,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.95
                }
            },
            timeout=20
        )

        if res1.status_code != 200:
            raise Exception(f"Ollama error {res1.status_code}")

        draft = res1.json().get("response", "").strip()

        print("\n🧠 DRAFT:\n", draft)

        if not draft:
            raise Exception("Draft vacío")

        # =========================
        # ✨ 2. REESCRITURA EDITORIAL
        # =========================
        prompt_refine = f"""
Actúa como editor senior de Farfetch.

Reescribe el siguiente texto para que sea:
- Más sofisticado
- Más fluido
- Con tono: {tone}
- 180–240 caracteres

Reglas:
- NO agregar información nueva

Texto:
{draft}

Salida:
Solo texto final.
"""

        res2 = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt_refine,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "top_p": 0.9
                }
            },
            timeout=20
        )

        final = res2.json().get("response", "").strip()

        print("\n✨ FINAL:\n", final)

        # =========================
        # 🔥 fallback inteligente (NO genérico)
        # =========================
        if not final:
            final = draft

        return re.sub(r"\s+", " ", final)[:260]

    except Exception as e:
        print("⚠️ Error Ollama:", e)

        # ⚠️ fallback SOLO si TODO falla
        base = f"{name} de {brand}".strip()

        if ctx["segment"] == "luxury":
            return f"{base} destaca por su ejecución precisa y materiales seleccionados, proyectando una estética refinada con carácter contemporáneo."

        if ctx["segment"] == "sport":
            return f"{base} ofrece una construcción ligera y dinámica, pensada para acompañar ritmos urbanos con energía y comodidad constante."

        return f"{base} se integra con naturalidad en el día a día, aportando una presencia equilibrada y versátil."

def build_details_array(details):
    result = []

    # 🔥 FEATURES
    for f in details.get("features", []):
        result.append(f"Característica: {f}")

    # 🔥 COMPOSICIÓN
    for c in details.get("composition", []):
        result.append(c)

    # 🔥 EXTRA
    for e in details.get("extra", []):
        if "Hecho en" in e:
            result.append(e)

    return result[:10]  # límite limpio

def normalize_record(record):
    fields = [
        "id",
        "sku",
        "externalRef",
        "ownerId",
        "name",
        "trade",
        "department",
        "price",
        "currency",
        "available",
        "description",
        "image",
        "images",
        "productType",
        "size",
        "quantity",
        "colors",
        "details",
        "source",
        "archived",
        "archivedAt",
        "archivedBy"
    ]

    normalized = {}

    for field in fields:
        normalized[field] = serialize_for_etl(record.get(field))

    return normalized

def serialize_for_etl(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return value


# =========================
# 
# =========================

# =========================
# EXCEL SEGURO
# =========================
def save_excel(records, filepath=None):
    import pandas as pd
    import time
    import os

    # 🔥 carpeta segura
    folder = "exports"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 🔥 nombre dinámico (evita PermissionError)
    if not filepath:
        filepath = os.path.join(folder, f"products_{int(time.time())}.xlsx")

    # 🔥 normalizar
    normalized = [normalize_record(r) for r in records]

    df = pd.DataFrame(normalized)

    # 🔥 ORDEN EXACTO (MODELO COMPLETO)
    columns = [
        "id",
        "sku",
        "externalRef",
        "ownerId",
        "name",
        "trade",
        "department",
        "price",
        "currency",
        "available",
        "description",
        "image",
        "images",
        "productType",
        "size",
        "quantity",
        "colors",
        "details",
        "source",
        "archived",
        "archivedAt",
        "archivedBy"
    ]

    # asegurar columnas
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df = df[columns]

    df.to_excel(filepath, index=False)

    return filepath
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
    save_excel(results, filepath="exports/partial.xlsx")

    return {
        "status": "ok",
        "products": len(results),
        "file": file,
        "data": results
    }
    
   
if __name__ == "__main__": run_crawler(max_workers=5)