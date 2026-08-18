import os
import re
import time
import json
import uuid
import random
import unicodedata
import requests
import pandas as pd

from fastapi import Body, FastAPI, HTTPException
from playwright.sync_api import sync_playwright

# =========================
# CONFIG
# =========================

USER_DATA_DIR = "./chrome_profile"

HEADLESS = False

MAX_PRODUCTS = 999

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17 Safari/605.1.15",
]

app = FastAPI(title="Farfetch Scraper Microservice")


@app.get("/")
def healthcheck():
    return {"status": "ok"}


@app.post("/scrape")
def scrape(urls: list[str] = Body(...)):
    if not urls:
        raise HTTPException(status_code=400, detail="Debes enviar al menos una URL")

    file, results = scrape_urls(urls[:MAX_PRODUCTS])

    return {
        "count": len(results),
        "export": file,
        "results": results,
    }

# =========================
# FOLDERS
# =========================

for folder in ["exports", "images"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# =========================
# HUMAN BEHAVIOR
# =========================

def human_delay(a=1.5, b=4):
    time.sleep(random.uniform(a, b))


def random_mouse(page):

    try:

        width = 1200
        height = 900

        for _ in range(random.randint(3, 8)):

            x = random.randint(50, width)
            y = random.randint(50, height)

            page.mouse.move(x, y)

            human_delay(0.1, 0.5)

    except:
        pass


def manual_scroll(page):

    try:

        for _ in range(random.randint(4, 10)):

            amount = random.randint(300, 1200)

            page.mouse.wheel(0, amount)

            human_delay(0.5, 2)

    except:
        pass


# =========================
# PLAYWRIGHT
# =========================

def create_browser():

    p = sync_playwright().start()

    browser = p.chromium.launch_persistent_context(
        USER_DATA_DIR,

        headless=HEADLESS,

        channel="chrome",

        viewport={"width": 1440, "height": 960},

        locale="es-CO",

        user_agent=random.choice(USER_AGENTS),

        args=[
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
            "--disable-dev-shm-usage",
            "--no-sandbox"
        ]
    )

    page = browser.new_page()

    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });

        window.chrome = {
            runtime: {}
        };

        Object.defineProperty(navigator, 'plugins', {
            get: () => [1,2,3,4,5]
        });

        Object.defineProperty(navigator, 'languages', {
            get: () => ['es-CO', 'es', 'en']
        });
    """)

    return p, browser, page


# =========================
# HELPERS
# =========================

def load_urls(filepath="products.txt"):

    if not os.path.exists(filepath):
        raise Exception(f"No existe {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        urls = [x.strip() for x in f.readlines() if x.strip()]

    return urls[:MAX_PRODUCTS]


def extract_ref(url):

    match = re.search(r'item-(\d+)', url)

    return match.group(1) if match else ""


def extract_department(url):

    match = re.search(r"/shopping/([^/]+)/", url)

    return match.group(1).lower() if match else ""


def normalize_text(text):

    text = text or ""

    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )


def clean_price(price_text):

    if not price_text:
        return 0

    cleaned = (
        price_text
        .replace("$", "")
        .replace(".", "")
        .replace(",", "")
        .replace("COP", "")
        .strip()
    )

    cleaned = re.sub(r"[^\d]", "", cleaned)

    try:
        return int(cleaned)
    except:
        return 0


def extract_price(page):

    selectors = [
        "[data-component='PriceFinal']",
        "[data-component='PriceOriginal']",
        "p:has-text('$')",
    ]

    for selector in selectors:
        try:
            prices = page.locator(selector).all()
            for price in prices:
                try:
                    text = price.inner_text().strip()
                    value = clean_price(text)
                    if value > 0:
                        return value
                except:
                    pass
        except:
            pass

    return 0


def clean_brand(brand):

    if not brand:
        return "BRAND"

    brand = re.sub(r"[^A-Za-z0-9]", "", brand.upper())

    return brand[:20]


def generate_sku(brand):

    brand_clean = clean_brand(brand)

    rand = uuid.uuid4().hex[:6]

    return f"{brand_clean}-{rand}"


def download_image(url, path):

    try:

        r = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": random.choice(USER_AGENTS)
            }
        )

        with open(path, "wb") as f:
            f.write(r.content)

        return True

    except:
        return False


# =========================
# DETAILS PARSER
# =========================
def extract_details(page):
    details = []

    try:
        # Abrir acordeón "Detalles del producto" si no está expandido
        accordions = page.locator("[data-component='AccordionButton']").all()
        for acc in accordions:
            try:
                text = acc.inner_text().lower()
                if "detalle" in text:
                    expanded = acc.get_attribute("aria-expanded")
                    if expanded != "true":
                        acc.click()
                        human_delay(1, 2)
                    break
            except:
                pass

        # =========================
        # CARACTERÍSTICAS (ul._fdc1e5 > li)
        # =========================
        feature_items = page.locator("ul._fdc1e5 li[data-component='Body']").all()
        for item in feature_items:
            try:
                text = item.inner_text().strip()
                if text and len(text) > 1:
                    details.append(text)
            except:
                pass

        # =========================
        # COMPOSICIÓN (busca el h4 "Composición" y toma el p siguiente)
        # =========================
        composition_blocks = page.locator(
            "[data-component='AccordionPanel'] h4[data-component='BodyBold']"
        ).all()
        for h4 in composition_blocks:
            try:
                label = h4.inner_text().strip().lower()
                if "composición" in label or "composicion" in label:
                    # El párrafo de composición está en el mismo div padre
                    parent = h4.locator("xpath=..") 
                    comp_text = parent.locator("p[data-component='Body']").first.inner_text().strip()
                    if comp_text:
                        details.append(f"Composición: {comp_text}")
                elif "medidas" in label:
                    parent = h4.locator("xpath=..")
                    medidas = parent.locator("p[data-component='Body']").all()
                    for m in medidas:
                        t = m.inner_text().strip()
                        if t:
                            details.append(t)
            except:
                pass

    except Exception as e:
        print("❌ details error:", e)

    # Remover duplicados manteniendo orden
    details = list(dict.fromkeys(details))
    return details[:30]
# =========================
# DESCRIPTION LLM STYLE
# =========================

def build_description(name, brand, details):

    text = ", ".join(details)

    if len(text) > 500:
        text = text[:500]

    description = (
        f"{name} de {brand} elaborado con detalles como "
        f"{text}. Diseño contemporáneo y acabado premium "
        f"para un look sofisticado y versátil."
    )

    return description[:1200]


# =========================
# CATEGORY
# =========================

def detect_category(name, details):

    text = f"{name} {' '.join(details)}".lower()

    rules = {
        "Tacones": ["tacón", "heel", "pump"],
        "Botas": ["bota", "boot"],
        "Mocasines": ["mocasín", "loafer"],
        "Mules": ["mule"],
        "Sandalias": ["sandalia"],
        "Bailarinas": ["bailarina"],
        "Chanclas": ["chancla"],
        "Zapatillas": ["tenis", "sneaker", "zapatilla"],
    }

    for cat, keys in rules.items():

        if any(k in text for k in keys):

            return cat

    return "Zapatillas"


# =========================
# SIZES
# =========================

def extract_sizes(page):

    sizes = []

    try:

        buttons = page.locator("button").all()

        for btn in buttons:

            try:

                text = btn.inner_text().strip()

                if re.search(r"\d+(\.\d+)?\s?(EU|US|UK|CM)", text.upper()):

                    sizes.append(text)

            except:
                pass

    except:
        pass

    return list(dict.fromkeys(sizes))


def extract_size_inventory(page):

    sizes = []
    quantities = []

    try:
        selector = page.locator("[data-testid='ScaledSizeSelector'] [role='combobox']").first
        if selector.count() > 0:
            selector.click()
            human_delay(0.5, 1)

        options = page.locator(
            "li[role='option'][data-component='SizeSelectorOption']"
        ).all()

        for option in options:
            try:
                option_text = option.inner_text().strip()

                size_text = ""
                try:
                    size_text = option.locator(
                        "[data-component='SizeSelectorOptionSize']"
                    ).first.inner_text().strip()
                except:
                    size_text = option_text

                match = re.search(r"\d+(?:\.\d+)?", size_text)
                if not match:
                    continue

                size = match.group(0)
                if size in sizes:
                    continue

                sizes.append(size)
                normalized = normalize_text(option_text)
                quantities.append(1 if "ultima pieza" in normalized else 2)

            except:
                pass

    except:
        pass

    if not sizes:
        sizes = extract_sizes(page)
        quantities = [2] * len(sizes)

    return sizes, quantities


# =========================
# IMAGES
# =========================

def extract_images(page, sku):

    images = []

    used = set()

    image_index = 1

    def save_large_image(src):
        nonlocal image_index

        if not src:
            return

        key = src.split("?")[0]

        if key in used:
            return

        used.add(key)

        filename = f"{sku}-{image_index}.jpg"

        path = f"images/{filename}"

        ok = download_image(src, path)

        if ok:
            images.append(path)
            image_index += 1

    try:

        selected_image = page.locator("#selected-image img").first

        thumbnails = page.locator(
            "button[data-carousel-item-button='true'][aria-label='Abrir imagen']"
        )

        count = thumbnails.count()

        for i in range(min(count, 5)):

            try:

                thumbnail = thumbnails.nth(i)

                thumbnail.scroll_into_view_if_needed(timeout=5000)

                previous_src = ""

                try:
                    previous_src = selected_image.get_attribute("src") or ""
                except:
                    pass

                thumbnail.click(force=True, timeout=10000)

                page.wait_for_selector("#selected-image img", timeout=10000)

                try:
                    page.wait_for_function(
                        """previous => {
                            const img = document.querySelector('#selected-image img');
                            return img && img.src && img.src !== previous;
                        }""",
                        previous_src,
                        timeout=5000
                    )
                except:
                    pass

                src = selected_image.get_attribute("src") or ""

                if "_300." in src:
                    src = src.replace("_300.", "_1000.")

                save_large_image(src)

                if len(images) >= 5:
                    break

                human_delay(0.4, 1)

            except:
                pass

    except:
        pass

    if images:
        return images

    try:

        img_elements = page.locator("img").all()

        for img in img_elements:

            try:

                src = img.get_attribute("src") or ""

                if (
                    "farfetch" in src.lower()
                    and (
                        ".jpg" in src.lower()
                        or ".jpeg" in src.lower()
                    )
                ):

                    if "_300." in src:
                        src = src.replace("_300.", "_1000.")

                    save_large_image(src)

                if len(images) >= 5:
                    break

            except:
                pass

    except:
        pass

    return images


# =========================
# SCRAPER
# =========================

def scrape_product(page, url):

    print(f"\n🔎 Opening:")
    print(url)

    try:

        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=90000
        )

        human_delay(4, 8)

        random_mouse(page)

        manual_scroll(page)

        page.wait_for_selector(
            "[data-testid='product-short-description']",
            timeout=30000
        )

        human_delay(1, 2)

        # =========================
        # NAME
        # =========================

        name = ""

        try:
            name = page.locator(
                "[data-testid='product-short-description']"
            ).first.inner_text().strip()
        except:
            pass

        # =========================
        # BRAND
        # =========================

        brand = "Unknown"

        try:
            brand = page.locator(
                "[data-ffref='pp_infobrd']"
            ).first.inner_text().strip()
        except:
            pass

        # =========================
        # PRICE
        # =========================

        price = 0

        try:

            price = extract_price(page)

        except:
            pass

        # =========================
        # SKU
        # =========================

        sku = generate_sku(brand)

        # =========================
        # IMAGES
        # =========================

        images = extract_images(page, sku)

        # =========================
        # DETAILS
        # =========================

        details = extract_details(page)

        # =========================
        # DESCRIPTION
        # =========================

        description = build_description(
            name,
            brand,
            details
        )

        # =========================
        # SIZES
        # =========================

        sizes, quantities = extract_size_inventory(page)

        # =========================
        # CATEGORY
        # =========================

        category = detect_category(
            name,
            details
        )

        # =========================
        # RESULT
        # =========================

        result = {

            "id": uuid.uuid4().hex[:8],

            "sku": sku,

            "externalRef": extract_ref(url),

            "ownerId": "9001",

            "name": name,

            "trade": brand,

            "department": extract_department(url),

            "price": price,

            "currency": "COP",

            "available": True,

            "description": description,

            "image": images[0] if images else "",

            "images": json.dumps(images, ensure_ascii=False),

            "productType": json.dumps([category], ensure_ascii=False),

            "category": category,

            "size": json.dumps(sizes, ensure_ascii=False),

            "quantity": json.dumps(quantities),

            "details": json.dumps(details, ensure_ascii=False),

            "source": url,

            "archived": False,

            "archivedAt": "",

            "archivedBy": ""

        }

        print("✅ OK:")
        print(name)

        return result

    except Exception as e:

        print("❌ ERROR:")
        print(e)

        return {
            "error": str(e),
            "url": url
        }


# =========================
# SAVE EXCEL
# =========================

def save_excel(records):

    filename = f"exports/products_{int(time.time())}.xlsx"

    df = pd.DataFrame(records)

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
        "details",
        "source",
        "archived",
        "archivedAt",
        "archivedBy"
    ]

    for col in columns:

        if col not in df.columns:
            df[col] = ""

    df = df[columns]

    df.to_excel(filename, index=False)

    return filename


# =========================
# MAIN
# =========================

def scrape_urls(urls):

    print(f"\n🚀 URLs: {len(urls)}")

    p, browser, page = create_browser()

    results = []

    try:

        # =========================
        # LOGIN MANUAL
        # =========================

        print("\n🌐 Opening homepage...")

        page.goto(
            "https://www.farfetch.com/",
            wait_until="domcontentloaded"
        )

        print("\n⏳ Tienes 10 segundos para:")
        print("- login")
        print("- captcha")
        print("- cookies")

        time.sleep(10)

        # =========================
        # SCRAPE
        # =========================

        for i, url in enumerate(urls, 1):

            print("\n=========================")
            print(f"[{i}/{len(urls)}]")
            print("=========================")

            result = scrape_product(page, url)

            results.append(result)

            # save incremental
            if len(results) % 3 == 0:

                file = save_excel(results)

                print("\n💾 Partial export:")
                print(file)

            human_delay(5, 12)

        # =========================
        # FINAL EXPORT
        # =========================

        file = save_excel(results)

        print("\n📦 FINAL EXPORT:")
        print(file)

        return file, results

    finally:

        browser.close()

        p.stop()


def run():

    urls = load_urls()

    scrape_urls(urls)


if __name__ == "__main__":

    run()
