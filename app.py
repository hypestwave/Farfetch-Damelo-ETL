import time
import uuid
import re
import os
import json
import base64
import requests
import pandas as pd
import cv2
import numpy as np

from fastapi import FastAPI
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from webdriver_manager.chrome import ChromeDriverManager
from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

EXCEL_COLUMNS = [
    "id", "sku", "externalRef", "ownerId", "name", "trade",
    "department", "price", "currency", "available", "description",
    "image", "images", "productType", "size", "quantity", "colors", "details",
]

if not os.path.exists("images"):
    os.makedirs("images")


# =========================
# DRIVER
# =========================
def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


# =========================
# DESCRIPCION — ESTRATEGIA 1: window.__initialState__
# =========================
def get_description_from_initial_state(driver):
    html = driver.page_source
    match = re.search(
        r'window\.__initialState__\s*=\s*(\{.*?\});\s*</script>',
        html, re.DOTALL
    )
    if not match:
        return {}
    try:
        state = json.loads(match.group(1))
        listing = (
            state.get("pdp", {})
            .get("productDetails", {})
            .get("listing", {})
        )
        if not listing:
            return {}
        details = [v for v in [
            listing.get("longDescription", ""),
            listing.get("composition", ""),
            listing.get("careInstructions", ""),
            listing.get("madeIn", ""),
            listing.get("heelHeight", ""),
            listing.get("designerNotes", ""),
        ] if v]
        return {
            "description": listing.get("shortDescription", ""),
            "details": details,
        }
    except (json.JSONDecodeError, KeyError):
        return {}


# =========================
# DESCRIPCION — ESTRATEGIA 2: JSON-LD
# =========================
def get_description_from_jsonld(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string)
            if isinstance(data, list):
                data = next((d for d in data if d.get("@type") == "Product"), {})
            if data.get("@type") == "Product":
                details = [v for v in [data.get("material", "")] if v]
                return {
                    "description": data.get("description", ""),
                    "details": details,
                }
        except (json.JSONDecodeError, AttributeError):
            continue
    return {}


# =========================
# DESCRIPCION — ESTRATEGIA 3: DOM (fallback)
# =========================
def get_description_from_dom(driver):
    selectors = [
        "[data-component='ProductDescription']",
        "[class*='productDescription']",
        "[class*='description']",
        "section[aria-label*='description' i]",
    ]
    for selector in selectors:
        try:
            el = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            text = el.text.strip()
            if len(text) > 40:
                return {"description": text, "details": []}
        except Exception:
            continue
    return {}


# =========================
# ORQUESTADOR DE DESCRIPCION
# =========================
def get_product_description(driver):
    empty = {"description": "", "details": []}
    for strategy in [
        get_description_from_initial_state,
        get_description_from_jsonld,
        get_description_from_dom,
    ]:
        result = strategy(driver)
        if result.get("description"):
            return result
    return empty


# =========================
# PRECIO
# =========================
def get_price(driver):
    try:
        wait = WebDriverWait(driver, 10)
        price_el = wait.until(
            EC.presence_of_element_located((
                By.XPATH,
                "//span[contains(text(),'$') or contains(text(),'COP')]"
            ))
        )
        price_text = price_el.text
        price_text = price_text.replace("$", "").replace(".", "").replace(",", "").strip()
        return float(price_text)
    except Exception:
        return get_price_from_html(driver)


def get_price_from_html(driver):
    html = driver.page_source
    match = re.search(r'"price":\s*([0-9]+)', html)
    if match:
        return float(match.group(1))
    return 0


# =========================
# IMAGENES (CAROUSEL) — slide-change-aware arrowRight
# Flujo por iteracion:
#   1. Leer src del slide visible actual
#   2. Descargar si es nueva
#   3. Click arrowRight (3 metodos en cascada)
#   4. Esperar hasta que el src del slide cambie (max 3 s)
# =========================

ARROW_SELECTORS = [
    # Selector exacto: data-flow-next + aria-controls="carousel-scroller"
    "//button[@data-flow-next='true' and @aria-controls='carousel-scroller']",
    # Variante sin aria-controls
    "//button[@data-flow-next='true']",
    # Atributo flow="next"
    "//button[@flow='next']",
    # aria-label en espanol (el sitio usa "Siguiente imagen del producto")
    "//button[contains(@aria-label,'Siguiente imagen')]",
    "//button[contains(@aria-label,'Siguiente')]",
    # Fallbacks genericos
    "//button[@aria-label='Next']",
    "//button[@aria-label='next']",
    "//button[@data-testid='carousel-next']",
    "//button[contains(@class,'arrow') and contains(@class,'right')]",
    "//button[contains(@class,'next')]",
    "//button[contains(@aria-label,'right') or contains(@aria-label,'Right')]",
]

def _get_active_src(driver) -> str:
    """
    Devuelve el src de la imagen del slide ACTUALMENTE VISIBLE en el carousel.
    El sitio usa #carousel-scroller como contenedor con scroll horizontal.
    La imagen visible es la primera que esta en el viewport segun getBoundingClientRect.
    """
    # Estrategia 1: JS — recorre los <img> dentro de #carousel-scroller
    # y devuelve el src de la primera que este dentro del viewport horizontal
    try:
        src = driver.execute_script("""
            var scroller = document.getElementById('carousel-scroller');
            if (!scroller) return null;
            var imgs = scroller.querySelectorAll('img');
            var sl = scroller.scrollLeft;
            var sw = scroller.clientWidth;
            for (var i = 0; i < imgs.length; i++) {
                var img = imgs[i];
                var rect = img.getBoundingClientRect();
                var srect = scroller.getBoundingClientRect();
                // visible si el centro horizontal esta dentro del scroller
                var centerX = rect.left + rect.width / 2;
                if (centerX >= srect.left && centerX <= srect.right) {
                    return img.src || img.getAttribute('data-src') || null;
                }
            }
            return imgs.length > 0 ? (imgs[0].src || imgs[0].getAttribute('data-src')) : null;
        """)
        if src and ("jpg" in src or "webp" in src or "png" in src):
            return src
    except Exception:
        pass

    # Estrategia 2: primer img grande dentro del scroller
    try:
        for img in driver.find_elements(By.XPATH, "//*[@id='carousel-scroller']//img"):
            src = img.get_attribute("src") or img.get_attribute("data-src") or ""
            if src and ("jpg" in src or "webp" in src or "png" in src):
                if img.size.get("width", 0) > 100:
                    return src
    except Exception:
        pass

    # Estrategia 3: clases active genericas
    for sel in [
        "//div[contains(@class,'active')]//img",
        "//li[contains(@class,'active')]//img",
        "//img[contains(@class,'active')]",
    ]:
        try:
            el = driver.find_element(By.XPATH, sel)
            src = el.get_attribute("src") or el.get_attribute("data-src") or ""
            if src and ("jpg" in src or "webp" in src or "png" in src):
                return src
        except Exception:
            continue

    # Fallback final: primera imagen grande del DOM
    try:
        for img in driver.find_elements(By.XPATH, "//img"):
            src = img.get_attribute("src") or img.get_attribute("data-src") or ""
            if src and ("jpg" in src or "webp" in src or "png" in src):
                if img.size.get("width", 0) > 200:
                    return src
    except Exception:
        pass

    return ""


def _click_arrow_right(driver) -> bool:
    """
    Localiza el boton arrowRight del carousel y lo activa con 3 metodos en cascada:
      1. ActionChains: hover + pausa + click (simula comportamiento humano)
      2. JS dispatchEvent MouseEvent bubbles (para listeners React/Vue)
      3. Tecla ArrowRight sobre el contenedor del carousel (ultimo recurso)
    Los selectores en ARROW_SELECTORS estan ordenados del mas especifico al mas generico,
    por lo que el primero que haga match es el correcto y los demas no se evaluan.
    """
    btn_el = None

    # Buscar el boton usando los selectores en orden de especificidad
    for sel in ARROW_SELECTORS:
        try:
            # find_element es instantaneo — no espera. Solo usamos WebDriverWait
            # para el primer selector (el exacto) donde sabemos que debe existir.
            btn_el = driver.find_element(By.XPATH, sel)
            break
        except Exception:
            continue

    if btn_el is None:
        # Ultimo recurso: ArrowRight sobre el scroller del carousel
        try:
            scroller = driver.find_element(By.ID, "carousel-scroller")
            scroller.send_keys(Keys.ARROW_RIGHT)
            return True
        except Exception:
            pass
        try:
            container = driver.find_element(
                By.XPATH,
                "//div[@id='carousel-scroller' or contains(@class,'carousel') "
                "or contains(@class,'slider') or contains(@class,'Carousel')]"
            )
            container.send_keys(Keys.ARROW_RIGHT)
            return True
        except Exception:
            return False

    # Metodo 1: ActionChains — hover + pausa breve + click
    try:
        ActionChains(driver).move_to_element(btn_el).pause(0.15).click().perform()
        return True
    except Exception:
        pass

    # Metodo 2: JS MouseEvent con bubbles (React/Next.js synthetic events)
    try:
        driver.execute_script(
            "arguments[0].dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));",
            btn_el
        )
        return True
    except Exception:
        pass

    # Metodo 3: click nativo del driver
    try:
        btn_el.click()
        return True
    except Exception:
        pass

    return False


def _wait_for_slide_change(driver, previous_src: str, timeout: float = 3.0) -> str:
    """
    Espera hasta `timeout` segundos a que el src del slide activo sea diferente
    de `previous_src`. Retorna el nuevo src (o el anterior si no cambio).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        new_src = _get_active_src(driver)
        if new_src and new_src != previous_src:
            return new_src
        time.sleep(0.2)
    return previous_src


def _download_image(src: str, filepath: str) -> bool:
    """Descarga `src` a `filepath`. Retorna True si tuvo exito."""
    try:
        r = requests.get(src, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def get_images(driver, product_id: str) -> list:
    """
    Captura exactamente 4 imagenes del carousel (o menos si el producto
    tiene menos slides). Cada imagen se guarda como:
        images/{product_id}_0.jpg
        images/{product_id}_1.jpg
        images/{product_id}_2.jpg
        images/{product_id}_3.jpg
    Retorna lista de rutas absolutas relativas al servidor (con / inicial).
    """
    local_paths = []
    seen_urls   = set()
    TARGET      = 4

    # Esperar a que el primer slide cargue
    time.sleep(2)

    current_src = _get_active_src(driver)

    for i in range(TARGET):
        # Si no hay src valido en este slide, intentar avanzar de todas formas
        if current_src and current_src not in seen_urls:
            seen_urls.add(current_src)
            filename = f"images/{product_id}_{i}.jpg"
            if _download_image(current_src, filename):
                local_paths.append("/" + filename)

        # En la ultima iteracion no hace falta avanzar
        if i == TARGET - 1:
            break

        # Click arrowRight
        clicked = _click_arrow_right(driver)

        if clicked:
            # Esperar cambio de slide real antes de la siguiente iteracion
            next_src = _wait_for_slide_change(driver, current_src, timeout=3.0)
            current_src = next_src
        else:
            # Sin boton de flecha: asumir carousel de 1 imagen, salir
            break

    return local_paths


# =========================
# SIZES
# =========================
def get_sizes(driver):
    try:
        btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'talla')]"))
        )
        btn.click()
        time.sleep(2)
        options = driver.find_elements(By.XPATH, "//li")
        sizes, quantity = [], []
        for opt in options:
            text = opt.text.strip()
            if text:
                sizes.append(text.split("\n")[0])
                quantity.append("1" if "Ultima pieza" in text else "2")
        if sizes:
            return sizes, quantity
    except Exception:
        pass
    return ["36", "37", "38"], ["2", "2", "2"]


# =========================
# NOMBRE DEL PRODUCTO (desde pagina)
# =========================
def get_product_name(driver) -> str:
    """
    Selector exacto del sitio:
    <p data-testid="product-short-description" data-component="Footnote">
      zapatillas con puntera cuadrada
    </p>
    """
    selectors = [
        (By.XPATH,       "//p[@data-testid='product-short-description']"),
        (By.CSS_SELECTOR,"[data-testid='product-short-description']"),
        (By.XPATH,       "//*[@data-component='Footnote' and @data-testid]"),
    ]
    for by, sel in selectors:
        try:
            el = WebDriverWait(driver, 6).until(
                EC.presence_of_element_located((by, sel))
            )
            text = el.text.strip()
            if text:
                return text
        except Exception:
            continue
    return ""


# =========================
# MARCA (desde pagina)
# =========================
def get_brand(driver):
    # Estrategia 1: enlace de marca exacto del sitio
    # <a data-ffref="pp_infobrd" data-component="LinkGhostDark" href="/co/shopping/women/{marca}/items.aspx">
    selectors = [
        "//a[@data-ffref='pp_infobrd']",
        "//a[@data-component='LinkGhostDark' and contains(@href,'/items.aspx')]",
        "//a[contains(@href,'/shopping/') and contains(@href,'/items.aspx')]",
    ]
    for sel in selectors:
        try:
            el = driver.find_element(By.XPATH, sel)
            text = el.text.strip()
            # Limpiar entidades HTML (ej. "Dolce &amp; Gabbana" -> "Dolce & Gabbana")
            from html import unescape
            text = unescape(text)
            if text:
                return text
        except Exception:
            continue

    # Estrategia 2: JSON-LD brand name
    html = driver.page_source
    match = re.search(r'"brand"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"', html)
    if match:
        from html import unescape
        return unescape(match.group(1))

    # Estrategia 3: window.__initialState__
    match = re.search(r'"designerName"\s*:\s*"([^"]+)"', html)
    if match:
        from html import unescape
        return unescape(match.group(1))

    return ""


# =========================
# VISION CV — COLOR DOMINANTE con OpenCV
# =========================
def analyze_image_opencv(image_path: str) -> dict:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"color_name": "unknown", "color_hex": "#000000"}
        img_small = cv2.resize(img, (100, 100))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        sorted_idx = np.argsort(-counts)
        dominant = None
        for idx in sorted_idx:
            r, g, b = centers[idx]
            if not (r > 220 and g > 220 and b > 220):
                dominant = (int(r), int(g), int(b))
                break
        if dominant is None:
            dominant = tuple(int(centers[sorted_idx[0]][i]) for i in range(3))
        r, g, b = dominant
        return {"color_name": rgb_to_color_name(r, g, b), "color_hex": f"#{r:02x}{g:02x}{b:02x}"}
    except Exception:
        return {"color_name": "unknown", "color_hex": "#000000"}


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    colors = {
        "negro":    (10,   10,  10),
        "blanco":   (245, 245, 245),
        "rojo":     (200,  30,  30),
        "azul":     ( 30,  60, 200),
        "verde":    ( 30, 150,  30),
        "amarillo": (230, 200,  20),
        "naranja":  (230, 120,  20),
        "rosa":     (230,  90, 150),
        "morado":   (120,  30, 180),
        "gris":     (130, 130, 130),
        "beige":    (210, 185, 155),
        "marron":   (130,  80,  30),
        "dorado":   (200, 165,  30),
        "plateado": (180, 180, 190),
    }
    best, min_dist = "indefinido", float("inf")
    for name, (cr, cg, cb) in colors.items():
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < min_dist:
            min_dist, best = dist, name
    return best


# =========================
# VISION IA — GPT-4o Vision
# Genera: description, details[], productType (texto libre), colors (multiples)
# =========================
def analyze_images_with_openai(image_paths: list, product_name: str, brand: str, existing_desc: str) -> dict:
    empty = {"ai_description": "", "ai_details": [], "ai_product_type": "", "ai_colors": ""}
    if not image_paths or not OPENAI_API_KEY:
        return empty

    image_contents = []
    for path in image_paths[:2]:
        try:
            with open(path.lstrip("/"), "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
        except Exception:
            continue

    if not image_contents:
        return empty

    prompt_text = f"""Eres un experto en moda de lujo. Analiza estas imagenes de producto.

Producto: {product_name}
Marca: {brand}
Descripcion existente (puede estar vacia): {existing_desc}

Responde SOLO con JSON valido, sin texto adicional ni bloques de codigo, con estas claves:

- "description": descripcion corta y atractiva en espanol (max 150 caracteres). Si ya existe descripcion, mejorala.
- "details": array de strings en espanol. Incluye: descripcion larga (1 frase), materiales visibles, cuidado sugerido, ocasion de uso, detalles de diseno notables. Minimo 3 items, maximo 6.
- "productType": categoria del producto en texto libre, en ingles, sin prefijos. Usa categorias como: "Pumps & Heels", "Sneakers", "Boots", "Ankle Boots", "Sandals", "Loafers", "Mules", "Flats", "Platforms", "Wedges", "Ballet Flats", "Bags", "Accessories". Puedes combinar con "|" si aplica mas de una (ej: "Pumps & Heels|Accessories").
- "colors": colores del producto visibles en las imagenes en formato "nombre:#hexcode", separados por "|" si hay varios (ej: "negro:#111111|dorado:#c9a84c").

Ejemplo de respuesta:
{{"description": "Zapatilla con puntera cuadrada en cuero negro", "details": ["Diseno sofisticado con puntera cuadrada y acabado en cuero liso negro", "Suela gruesa de goma con contraste blanco", "Tacon bajo de 3 cm", "Ideal para looks urbanos y formales", "Cierre con cordones planos"], "productType": "Sneakers", "colors": "negro:#1c1c1c|blanco:#f5f5f5"}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": image_contents + [{"type": "text", "text": prompt_text}]
            }]
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
        parsed = json.loads(raw)
        return {
            "ai_description":  parsed.get("description", ""),
            "ai_details":      parsed.get("details", []),
            "ai_product_type": parsed.get("productType", ""),
            "ai_colors":       parsed.get("colors", ""),
        }
    except Exception:
        return empty


# =========================
# SCRAPER PRINCIPAL
# =========================
def scrape(url):
    driver = create_driver()
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

    # trade: <a data-ffref="pp_infobrd">Dolce & Gabbana</a>
    brand = get_brand(driver)

    # name: <p data-testid="product-short-description">zapatillas con puntera cuadrada</p>
    name = get_product_name(driver)
    if not name:
        h1_elements = driver.find_elements(By.TAG_NAME, "h1")
        if len(h1_elements) >= 2:
            name = h1_elements[1].text.strip()
        elif len(h1_elements) == 1:
            name = h1_elements[0].text.strip()
        else:
            name = brand
    price = get_price(driver)

    product_info = get_product_description(driver)
    description  = product_info["description"]
    details      = product_info["details"]

    product_id   = str(uuid.uuid4())[:8]
    local_images = get_images(driver, product_id)
    sizes, quantity = get_sizes(driver)

    driver.quit()

    # OpenCV: color dominante (fallback)
    cv_color = {"color_name": "unknown", "color_hex": "#000000"}
    if local_images:
        cv_color = analyze_image_opencv(local_images[0].lstrip("/"))

    # GPT-4o Vision
    ai_data = analyze_images_with_openai(local_images, name, brand, description)

    if not description and ai_data.get("ai_description"):
        description = ai_data["ai_description"]

    if not details and ai_data.get("ai_details"):
        details = ai_data["ai_details"]

    product_type = ai_data.get("ai_product_type") or _infer_product_type(name)

    colors_str = (
        ai_data.get("ai_colors")
        or f"{cv_color['color_name']}:{cv_color['color_hex']}"
    )

    sku = f"{brand.replace(' ', '-').upper()}-{product_id}" if brand else f"XX-{product_id}"
    ext_match = re.search(r'item-(\d+)', url) or re.search(r'-(\d+)\.aspx', url)
    ext_ref = ext_match.group(1) if ext_match else ""

    return {
        "id":          product_id,
        "sku":         sku,
        "externalRef": ext_ref,
        "ownerId":     "9001",
        "name":        name,
        "trade":       brand,
        "department":  "Women",
        "price":       price,
        "currency":    "COP",
        "available":   True,
        "description": description,
        "image":       local_images[0] if local_images else "",
        "images":      "|".join(local_images),
        "productType": product_type,
        "size":        "|".join(sizes),
        "quantity":    "|".join([str(q) for q in quantity]),
        "colors":      colors_str,
        "details":     "|".join(details),
    }


def _infer_product_type(name: str) -> str:
    n = name.lower()
    mapping = [
        ("sneaker",    "Sneakers"),
        ("zapatilla",  "Sneakers"),
        ("tenis",      "Sneakers"),
        ("tacón",      "Pumps & Heels"),
        ("tacon",      "Pumps & Heels"),
        ("heel",       "Pumps & Heels"),
        ("pump",       "Pumps & Heels"),
        ("ankle boot", "Ankle Boots"),
        ("botin",      "Ankle Boots"),
        ("botín",      "Ankle Boots"),
        ("bota",       "Boots"),
        ("boot",       "Boots"),
        ("sandalia",   "Sandals"),
        ("sandal",     "Sandals"),
        ("loafer",     "Loafers"),
        ("mocasin",    "Loafers"),
        ("mule",       "Mules"),
        ("plataforma", "Platforms"),
        ("platform",   "Platforms"),
        ("wedge",      "Wedges"),
        ("ballet",     "Ballet Flats"),
        ("flat",       "Flats"),
        ("bolso",      "Bags"),
        ("bag",        "Bags"),
    ]
    for keyword, ptype in mapping:
        if keyword in n:
            return ptype
    return "Shoes"


# =========================
# GUARDAR EXCEL — columnas canonicas + formato profesional
# =========================
def save_to_excel(records: list, filepath: str = "products.xlsx"):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Products"

    header_font  = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    header_fill  = PatternFill("solid", start_color="1A1A2E")
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align   = Alignment(horizontal="left",   vertical="center", wrap_text=True)
    thin = Side(style="thin", color="CCCCCC")
    thin_border  = Border(left=thin, right=thin, top=thin, bottom=thin)
    alt_fill     = PatternFill("solid", start_color="F5F5F5")

    ws.append(EXCEL_COLUMNS)
    for col_idx in range(1, len(EXCEL_COLUMNS) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = center_align
        cell.border    = thin_border

    for row_idx, record in enumerate(records, start=2):
        ws.append([record.get(col, "") for col in EXCEL_COLUMNS])
        use_alt = row_idx % 2 == 0
        for col_idx in range(1, len(EXCEL_COLUMNS) + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border    = thin_border
            cell.alignment = left_align
            cell.font      = Font(name="Arial", size=10)
            if use_alt:
                cell.fill = alt_fill

    col_widths = {
        "id": 12, "sku": 28, "externalRef": 18, "ownerId": 10,
        "name": 35, "trade": 22, "department": 14, "price": 12,
        "currency": 10, "available": 11, "description": 50,
        "image": 30, "images": 40, "productType": 25,
        "size": 20, "quantity": 20, "colors": 30, "details": 60,
    }
    for col_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 20)

    ws.row_dimensions[1].height = 22
    ws.freeze_panes = "A2"

    wb.save(filepath)
    return filepath


# =========================
# ENDPOINT
# =========================
@app.post("/scrape")
def scrape_products():
    url = "https://www.productsite.com/co/shopping/women/dolce-gabbana-zapatillas-con-puntera-cuadrada-item-33756476.aspx"

    data = scrape(url)
    save_to_excel([data], "products.xlsx")

    return {
        "status": "ok",
        "images_saved": data["images"],
        "file": "products.xlsx",
    }
