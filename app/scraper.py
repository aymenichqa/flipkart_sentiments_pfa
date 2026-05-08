# app/scraper.py — Moteur de Scraping V4
# ══════════════════════════════════════════════════════════════════════════════
#
# Sources supportées :
#   JumiaScraper        → jumia.ma          (AJAX HTML fragments)
#   GoogleMapsScraper   → maps.app.goo.gl   (infinite scroll)
#   TrustpilotScraper   → trustpilot.com    (__NEXT_DATA__ JSON + HTML)
#   TripAdvisorScraper  → tripadvisor.com   (data-automation cards)
#
# Technologie : sync_playwright pour éviter le conflit event-loop avec FastAPI.
#
# Prérequis :
#   pip install playwright beautifulsoup4 lxml requests
#   playwright install chromium
# ══════════════════════════════════════════════════════════════════════════════

import re
import json
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Dict

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass commun
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScrapedReview:
    text   : str
    rating : Optional[float]
    source : str
    target : str
    date   : Optional[str] = None
    langue : Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "text"  : self.text,
            "rating": self.rating,
            "source": self.source,
            "target": self.target,
            "date"  : self.date,
            "langue": self.langue,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Classe de base Playwright
# ─────────────────────────────────────────────────────────────────────────────

class BasePlaywrightScraper:

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def _browser_context_args(self, locale: str = "fr-MA") -> dict:
        return dict(
            user_agent         = self.USER_AGENT,
            viewport           = {"width": 1440, "height": 900},
            locale             = locale,
            timezone_id        = "Africa/Casablanca",
            extra_http_headers = {
                "Accept-Language": "fr-MA,fr;q=0.9,ar;q=0.8,en;q=0.7",
            },
        )

    def _launch_page(self, pw, locale="fr-MA"):
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = browser.new_context(**self._browser_context_args(locale))
        page    = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return browser, page

    def _soup(self, page) -> BeautifulSoup:
        return BeautifulSoup(page.content(), "lxml")


# ─────────────────────────────────────────────────────────────────────────────
# JumiaScraper — jumia.ma
# ─────────────────────────────────────────────────────────────────────────────
#
# URL type : https://www.jumia.ma/catalog/productratingsreviews/sku/SKU/
#
# CORRECTIF V4 :
#   - Le vrai sélecteur est  article[class*="rv"]  et NON  article.article
#   - Le texte est dans      p.rvw-desc  ou  .rv-bd p
#   - La note est encodée dans la classe CSS :  _stars-4  → 4/5
#   - L'endpoint AJAX retourne du HTML pur → on le charge avec Playwright
#     puis on récupère le contenu avec un wait adapté

class JumiaScraper(BasePlaywrightScraper):

    def scrape(self, url: str, max_pages: int = 3) -> List[ScrapedReview]:
        print(f"\n🛍️  JumiaScraper — {url}")
        reviews = []

        # Normaliser l'URL : on veut toujours l'endpoint reviews
        base = url.rstrip("/")
        if "/productratingsreviews/" not in base:
            base = base + "/reviews"

        with sync_playwright() as pw:
            browser, page = self._launch_page(pw)
            try:
                for page_num in range(1, max_pages + 1):
                    page_url = f"{base}/?page={page_num}"
                    print(f"  📄 Page {page_num}/{max_pages} → {page_url}")

                    try:
                        page.goto(page_url, wait_until="domcontentloaded", timeout=30_000)
                    except PWTimeout:
                        print(f"  ⏰ Timeout chargement page {page_num}")
                        break

                    # Attendre que les articles soient présents
                    try:
                        page.wait_for_selector("article[class*='rv']", timeout=12_000)
                    except PWTimeout:
                        print(f"  ⚠️  Aucun article[class*='rv'] — tentative sélecteur de secours")

                    # Pause pour le rendu JS
                    page.wait_for_timeout(1_500)
                    soup = self._soup(page)

                    cards = self._find_cards(soup)
                    if not cards:
                        print(f"  ℹ️  Aucune carte d'avis page {page_num} — fin")
                        break

                    before = len(reviews)
                    for card in cards:
                        r = self._parse_card(card, url)
                        if r:
                            reviews.append(r)

                    added = len(reviews) - before
                    print(f"  ✅ {added} avis extraits (page {page_num})")

                    if added == 0:
                        break

                    page.wait_for_timeout(random.randint(1_200, 2_500))

            finally:
                browser.close()

        print(f"  🎯 Total : {len(reviews)} avis Jumia")
        return reviews if reviews else self._fallback(url)

    def _find_cards(self, soup: BeautifulSoup):
        """
        Sélecteurs en cascade — du plus précis au plus large.
        Corrects pour la structure réelle de Jumia MA (2024-2025) :
            <article class="rv rv--sm col12"> … </article>
        """
        for sel in [
            "article[class*='rv']",          # ✅ sélecteur principal corrigé
            "article.rv",                    # variante directe
            "div.rv-bd",                     # corps de l'avis seul
            "div[class*='rv-bd']",
            "article",                       # fallback large
        ]:
            cards = soup.select(sel)
            valid = [c for c in cards if len(c.get_text(strip=True)) > 10]
            if valid:
                print(f"  🔍 Sélecteur actif : '{sel}' ({len(valid)} blocs)")
                return valid
        return []

    def _parse_card(self, card: BeautifulSoup, url: str) -> Optional[ScrapedReview]:
        # Texte — sélecteurs dans l'ordre de probabilité
        text = None
        for sel in ["p.rvw-desc", "p[class*='desc']", ".rv-bd p", "p"]:
            el = card.select_one(sel)
            if el:
                t = el.get_text(strip=True)
                if len(t) > 8:
                    text = t
                    break

        if not text:
            # Nettoyage et fallback sur le texte brut de la carte
            for tag in card.find_all(["button", "img", "svg"]):
                tag.decompose()
            text = card.get_text(separator=" ", strip=True)
            if len(text) < 8:
                return None

        # Note — encodée dans la classe CSS : "_stars-4" → 4
        rating = None
        stars_el = card.select_one("div[class*='stars'], span[class*='stars']")
        if stars_el:
            cls = " ".join(stars_el.get("class", []))
            # Format : _stars-4  (Jumia MA)
            m = re.search(r"_stars-(\d)", cls)
            if m:
                rating = float(m.group(1))
            else:
                # Format alternatif : _4 seul
                m2 = re.search(r"\s_(\d)(?:\s|$)", " " + cls + " ")
                if m2:
                    v = float(m2.group(1))
                    rating = v if v <= 5 else v / 10

        # Date
        date = None
        d_el = card.select_one("span.rvw-date, time, [class*='date']")
        if d_el:
            date = d_el.get("datetime", d_el.get_text(strip=True))

        return ScrapedReview(
            text=text, rating=rating,
            source="Jumia Maroc", target=url, date=date, langue="fr",
        )

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Fallback Jumia (données démo)")
        data = [
            ("Produit reçu rapidement, qualité conforme. Je recommande !", 5.0),
            ("Bon rapport qualité-prix. Livraison en 3 jours.", 4.0),
            ("Correct, rien d'exceptionnel. Emballage un peu abîmé.", 3.0),
            ("Déçu. Qualité décevante par rapport aux photos.", 2.0),
            ("Arnaque ! Produit complètement différent.", 1.0),
            ("Mazyan bzzaf had l'produit, ghir l packaging kayn chi mochkil.", 4.0),
        ]
        return [ScrapedReview(t, r, "Jumia (Démo)", url, langue="fr") for t, r in data]


# ─────────────────────────────────────────────────────────────────────────────
# GoogleMapsScraper — maps.app.goo.gl / google.com/maps
# ─────────────────────────────────────────────────────────────────────────────
#
# CORRECTIFS V4 :
#   - Gestion des URLs courtes (maps.app.goo.gl) : on attend la redirection
#   - Sélecteur scroll : div[role="feed"] en plus de .m6QErb
#   - Bouton "Avis" : plusieurs tentatives avec waits adaptatifs
#   - "Voir plus" : déplier avant extraction
#   - Sélecteur texte : wiI7pd (actuel en 2025)

class GoogleMapsScraper(BasePlaywrightScraper):

    # Sélecteurs du conteneur scrollable (changent régulièrement chez Google)
    FEED_SELECTORS = [
        "div[role='feed']",           # ✅ le plus stable en 2025
        "div.m6QErb[data-tab-index]", # variante avec attribut
        "div.m6QErb",                 # ancien sélecteur
        "div.section-scrollbox",      # très ancien
    ]

    # Sélecteurs pour les cartes individuelles
    CARD_SELECTORS = [
        "div[data-review-id]",        # ✅ le plus stable
        "div.jftiEf",
        "div[class*='jftiEf']",
        "div.GHT2ce",
    ]

    # Sélecteurs texte
    TEXT_SELECTORS = [
        "span.wiI7pd",                # ✅ actuel 2025
        "[class*='wiI7pd']",
        "span[data-expanded-text]",
        "span.MyEned span",
        ".review-full-text",
    ]

    def scrape(self, url: str, target_reviews: int = 20) -> List[ScrapedReview]:
        print(f"\n📍 GoogleMapsScraper — {url}")
        print(f"  🎯 Objectif : {target_reviews} avis")

        reviews_data = []

        with sync_playwright() as pw:
            browser, page = self._launch_page(pw, locale="fr-FR")
            try:
                # ── Chargement (supporte les URLs courtes goo.gl) ────────────
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                except PWTimeout:
                    print("  ⏰ Timeout chargement page")
                    browser.close()
                    return self._fallback(url)

                # Attendre la fin des redirections
                page.wait_for_timeout(4_000)
                print(f"  🌐 URL réelle : {page.url[:80]}...")

                # ── Cliquer sur l'onglet "Avis" ──────────────────────────────
                self._click_reviews_tab(page)

                # ── Trouver et scroller le feed ──────────────────────────────
                feed_sel = self._find_feed_selector(page)
                if feed_sel:
                    self._scroll_feed(page, feed_sel, target_reviews)
                else:
                    print("  ⚠️  Feed non trouvé — tentative scroll fenêtre")
                    for _ in range(target_reviews // 3):
                        page.keyboard.press("End")
                        page.wait_for_timeout(1_500)

                # ── Déplier les avis tronqués ────────────────────────────────
                self._expand_reviews(page)

                # ── Extraire ─────────────────────────────────────────────────
                soup = self._soup(page)
                reviews_data = self._extract_reviews(soup, url)

            finally:
                browser.close()

        print(f"  🎯 Total : {len(reviews_data)} avis Google Maps")
        return reviews_data if reviews_data else self._fallback(url)

    def _click_reviews_tab(self, page):
        """Cherche et clique sur le bouton 'Avis' avec plusieurs stratégies."""
        selectors = [
            "button[aria-label*='Avis']",
            "button[aria-label*='Reviews']",
            "button[aria-label*='avis']",
            "//button[contains(., 'Avis')]",
            "//button[contains(., 'Reviews')]",
            "[data-tab-index='1']",
        ]
        for sel in selectors:
            try:
                if sel.startswith("//"):
                    loc = page.locator(f"xpath={sel}").first
                else:
                    loc = page.locator(sel).first
                if loc.is_visible(timeout=2_000):
                    loc.click()
                    print("  ✅ Onglet 'Avis' cliqué")
                    page.wait_for_timeout(3_000)
                    return
            except Exception:
                continue
        print("  ⚠️  Onglet 'Avis' non trouvé (peut-être déjà visible)")

    def _find_feed_selector(self, page) -> Optional[str]:
        """Retourne le premier sélecteur de feed qui existe sur la page."""
        for sel in self.FEED_SELECTORS:
            try:
                page.wait_for_selector(sel, timeout=5_000)
                print(f"  📦 Feed trouvé : '{sel}'")
                return sel
            except PWTimeout:
                continue
        return None

    def _scroll_feed(self, page, feed_sel: str, target: int):
        """Scroll dans le div feed jusqu'à target avis ou blocage."""
        # Script pour scroller dans la div (pas window)
        scroll_js = f"""
            (() => {{
                const el = document.querySelector("{feed_sel}");
                if (el) {{ el.scrollTop += 900; return el.scrollTop; }}
                return -1;
            }})()
        """
        prev_count = 0
        stale      = 0
        max_iter   = max(target // 4, 15)

        for i in range(max_iter):
            # Compter les cartes actuellement chargées
            count = 0
            for csel in self.CARD_SELECTORS:
                els = page.query_selector_all(csel)
                if els:
                    count = len(els)
                    break

            if count >= target:
                print(f"  🎯 {count} avis chargés — objectif atteint")
                break

            if count == prev_count:
                stale += 1
                if stale >= 4:
                    print(f"  ℹ️  Fin du feed — {count} avis au total")
                    break
            else:
                stale = 0
                print(f"  📜 Scroll {i+1} — {count} avis chargés…")

            prev_count = count
            page.evaluate(scroll_js)
            page.wait_for_timeout(random.randint(1_200, 2_000))

    def _expand_reviews(self, page):
        """Déplie les avis tronqués ('Voir plus' / 'See more')."""
        for sel in [
            "button[aria-label*='Voir plus']",
            "button[aria-label*='See more']",
            "button[aria-label*='voir plus']",
            "button.w8nwRe",
            "[jsaction*='review-full-text']",
        ]:
            try:
                btns = page.query_selector_all(sel)
                for btn in btns[:20]:
                    try:
                        btn.click()
                        page.wait_for_timeout(200)
                    except Exception:
                        pass
                if btns:
                    print(f"  📖 {len(btns)} avis dépliés")
                    break
            except Exception:
                continue

    def _extract_reviews(self, soup: BeautifulSoup, url: str) -> List[ScrapedReview]:
        reviews = []

        # Trouver les cartes
        cards = []
        for sel in self.CARD_SELECTORS:
            cards = soup.select(sel)
            if cards:
                print(f"  🔍 Sélecteur carte : '{sel}' ({len(cards)} cartes)")
                break

        if not cards:
            print("  ⚠️  Aucune carte d'avis dans le HTML final")
            return []

        for card in cards:
            # Texte
            text = None
            for sel in self.TEXT_SELECTORS:
                el = card.select_one(sel)
                if el:
                    t = el.get_text(strip=True)
                    if len(t) > 5:
                        text = t
                        break
            if not text:
                continue

            # Note
            rating = None
            r_el = card.select_one("span[role='img'][aria-label]")
            if r_el:
                aria = r_el.get("aria-label", "")
                m = re.search(r"(\d+(?:[.,]\d+)?)", aria)
                if m:
                    rating = float(m.group(1).replace(",", "."))
                    if rating > 5:
                        rating = rating / 10

            # Date
            date = None
            d_el = card.select_one("[class*='rsqaWe'], span[jsan*='date']")
            if d_el:
                date = d_el.get_text(strip=True)

            reviews.append(ScrapedReview(
                text=text, rating=rating,
                source="Google Maps", target=url,
                date=date, langue="fr",
            ))

        return reviews

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Fallback Google Maps (données démo)")
        data = [
            ("Excellent service, personnel très accueillant. Je reviendrai !", 5.0),
            ("Bonne ambiance, cuisine délicieuse et prix raisonnables.", 5.0),
            ("Service correct mais l'attente est un peu longue.", 3.0),
            ("Déçu par le service. Personnel peu aimable.", 2.0),
            ("Endroit mezian, walakin les prix 3alin chwiya.", 4.0),
            ("Très bonne expérience globale. Je recommande.", 5.0),
        ]
        return [ScrapedReview(t, r, "Google Maps (Démo)", url, langue="fr") for t, r in data]


# ─────────────────────────────────────────────────────────────────────────────
# TrustpilotScraper — trustpilot.com
# ─────────────────────────────────────────────────────────────────────────────
#
# URL type : https://www.trustpilot.com/review/amazon.in
#
# STRATÉGIE :
#   Trustpilot est une app Next.js. Les données sont dans __NEXT_DATA__ (JSON)
#   embedé dans le HTML.
#   Playwright charge la page → on extrait window.__NEXT_DATA__ via JS eval
#   → on parse les reviews directement depuis le JSON (aucun CSS selector fragile).
#   Fallback : sélecteurs HTML article[data-service-review-card-paper]

class TrustpilotScraper(BasePlaywrightScraper):

    RATING_MAP = {
        "5": 5.0, "4": 4.0, "3": 3.0, "2": 2.0, "1": 1.0,
        "bad": 1.0, "poor": 2.0, "average": 3.0, "great": 4.0, "excellent": 5.0,
    }

    def scrape(self, url: str, max_pages: int = 3) -> List[ScrapedReview]:
        print(f"\n⭐ TrustpilotScraper — {url}")
        reviews = []

        with sync_playwright() as pw:
            browser, page = self._launch_page(pw, locale="fr-FR")
            try:
                for page_num in range(1, max_pages + 1):
                    page_url = url if page_num == 1 else f"{url.rstrip('/')}?page={page_num}"
                    print(f"  📄 Page {page_num}/{max_pages}")

                    try:
                        page.goto(page_url, wait_until="domcontentloaded", timeout=35_000)
                    except PWTimeout:
                        print(f"  ⏰ Timeout page {page_num}")
                        break

                    # Attendre les reviews ou __NEXT_DATA__
                    page.wait_for_timeout(3_000)

                    # ── Stratégie 1 : extraire __NEXT_DATA__ (JSON) ───────────
                    batch = self._extract_next_data(page, url)
                    if batch:
                        reviews.extend(batch)
                        print(f"  ✅ {len(batch)} avis via __NEXT_DATA__")
                    else:
                        # ── Stratégie 2 : HTML classique ─────────────────────
                        batch = self._extract_html(page, url)
                        reviews.extend(batch)
                        print(f"  ✅ {len(batch)} avis via HTML")

                    if not batch:
                        print(f"  ℹ️  Aucun avis page {page_num} — arrêt")
                        break

                    page.wait_for_timeout(random.randint(1_500, 2_500))

            finally:
                browser.close()

        print(f"  🎯 Total : {len(reviews)} avis Trustpilot")
        return reviews if reviews else self._fallback(url)

    def _extract_next_data(self, page, url: str) -> List[ScrapedReview]:
        """Extrait les reviews depuis window.__NEXT_DATA__ (méthode la plus fiable)."""
        try:
            # Extraire via JavaScript eval
            data_str = page.evaluate("() => JSON.stringify(window.__NEXT_DATA__)")
            if not data_str:
                return []

            data = json.loads(data_str)

            # Naviguer vers les reviews dans le JSON Next.js
            reviews_list = self._deep_find_reviews(data)
            if not reviews_list:
                return []

            results = []
            for rv in reviews_list:
                text   = rv.get("text", rv.get("reviewBody", "")).strip()
                if len(text) < 5:
                    continue

                rating = None
                stars  = rv.get("stars", rv.get("rating", rv.get("ratingValue")))
                if stars is not None:
                    try:
                        rating = float(str(stars))
                        if rating > 5:
                            rating = rating / 10
                    except (ValueError, TypeError):
                        pass

                date = rv.get("createdAt", rv.get("datePublished", ""))
                if isinstance(date, str) and "T" in date:
                    date = date.split("T")[0]

                results.append(ScrapedReview(
                    text=text, rating=rating,
                    source="Trustpilot", target=url,
                    date=date, langue="fr",
                ))
            return results

        except Exception as e:
            print(f"  ⚠️  __NEXT_DATA__ non parseable : {e}")
            return []

    def _deep_find_reviews(self, obj, depth=0) -> Optional[list]:
        """Cherche récursivement une liste de reviews dans le JSON Next.js."""
        if depth > 8:
            return None
        if isinstance(obj, dict):
            for key in ("reviews", "reviewsList", "pageData"):
                if key in obj and isinstance(obj[key], list) and len(obj[key]) > 0:
                    if isinstance(obj[key][0], dict) and any(
                        k in obj[key][0] for k in ("text", "reviewBody", "stars", "rating")
                    ):
                        return obj[key]
            for v in obj.values():
                result = self._deep_find_reviews(v, depth + 1)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj[:5]:
                result = self._deep_find_reviews(item, depth + 1)
                if result:
                    return result
        return None

    def _extract_html(self, page, url: str) -> List[ScrapedReview]:
        """Fallback HTML — sélecteurs Trustpilot (Next.js génère des classes stables via data-attributes)."""
        soup = self._soup(page)
        results = []

        # Sélecteurs de cartes (stable via data-attributes)
        card_selectors = [
            "article[data-service-review-card-paper]",
            "div[data-service-review-card-paper]",
            "article.paper_paper",
            "[class*='reviewCard']",
            "article",
        ]
        cards = []
        for sel in card_selectors:
            c = [x for x in soup.select(sel) if len(x.get_text(strip=True)) > 20]
            if c:
                print(f"  🔍 HTML sélecteur : '{sel}' ({len(c)} blocs)")
                cards = c
                break

        for card in cards:
            # Texte — Trustpilot met le texte dans un <p> avec data-service-review-text
            text = None
            for sel in [
                "[data-service-review-text-typography]",
                "p[class*='typography_body']",
                "[class*='review-content']",
                "p",
            ]:
                el = card.select_one(sel)
                if el:
                    t = el.get_text(strip=True)
                    if len(t) > 10:
                        text = t
                        break
            if not text:
                continue

            # Note — Trustpilot encode la note dans img[alt] ou data-rating
            rating = None
            img_el = card.select_one("img[alt*='étoile'], img[alt*='star'], img[alt*='Rated']")
            if img_el:
                m = re.search(r"(\d+(?:\.\d+)?)", img_el.get("alt", ""))
                if m:
                    rating = float(m.group(1))
            if rating is None:
                # Chercher un attribut data-rating
                r_el = card.find(attrs={"data-rating": True})
                if r_el:
                    try:
                        rating = float(r_el["data-rating"])
                    except (ValueError, TypeError):
                        pass

            # Date
            date = None
            time_el = card.select_one("time[datetime]")
            if time_el:
                date = time_el.get("datetime", "")[:10]

            results.append(ScrapedReview(
                text=text, rating=rating,
                source="Trustpilot", target=url,
                date=date, langue="en",
            ))

        return results

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Fallback Trustpilot (données démo)")
        data = [
            ("Exceptional service! Very fast delivery and the product quality exceeded my expectations.", 5.0),
            ("Great experience overall. The customer support was responsive and helpful.", 5.0),
            ("Average experience. Nothing outstanding but nothing terrible either.", 3.0),
            ("Disappointed with the quality. The item looked different from the pictures.", 2.0),
            ("Terrible experience. Never received my order and no response from support.", 1.0),
            ("Really solid company. I've ordered multiple times without issues.", 4.0),
        ]
        return [ScrapedReview(t, r, "Trustpilot (Démo)", url, langue="en") for t, r in data]


# ─────────────────────────────────────────────────────────────────────────────
# TripAdvisorScraper — tripadvisor.com
# ─────────────────────────────────────────────────────────────────────────────
#
# URL type : https://www.tripadvisor.com/Restaurant_Review-g293732-d24178657-Reviews-...
#
# STRATÉGIE :
#   TripAdvisor charge les avis via JS.
#   On utilise Playwright + data-automation="reviewCard" pour cibler les cartes.
#   Pagination via le bouton "Next" ou via paramètre URL (#REVIEWS-offset).

class TripAdvisorScraper(BasePlaywrightScraper):

    CARD_SELECTORS = [
        "[data-automation='reviewCard']",       # ✅ le plus stable
        "div[class*='reviewCard']",
        "div.biGQs",
        "div[class*='listCell']",
    ]

    TEXT_SELECTORS = [
        "span.yCeTE",                           # ✅ texte avis 2025
        "[data-automation='reviewText'] span",
        "[class*='reviewText']",
        "[class*='content'] span",
        "q span",
        "span[class*='yCeTE']",
    ]

    def scrape(self, url: str, max_pages: int = 3) -> List[ScrapedReview]:
        print(f"\n🗺️  TripAdvisorScraper — {url}")
        reviews = []

        with sync_playwright() as pw:
            browser, page = self._launch_page(pw, locale="fr-FR")
            try:
                for page_num in range(1, max_pages + 1):
                    page_url = self._page_url(url, page_num)
                    print(f"  📄 Page {page_num}/{max_pages}")

                    try:
                        page.goto(page_url, wait_until="domcontentloaded", timeout=40_000)
                    except PWTimeout:
                        print(f"  ⏰ Timeout page {page_num}")
                        break

                    # Fermer popups éventuels
                    self._close_popups(page)
                    page.wait_for_timeout(3_000)

                    # Attendre au moins une carte d'avis
                    found = False
                    for sel in self.CARD_SELECTORS:
                        try:
                            page.wait_for_selector(sel, timeout=10_000)
                            found = True
                            break
                        except PWTimeout:
                            continue

                    if not found:
                        print(f"  ⚠️  Aucune carte page {page_num}")
                        break

                    # Déplier les "Voir plus" / "Plus"
                    self._expand_reviews(page)

                    soup  = self._soup(page)
                    batch = self._extract_reviews(soup, url)

                    if not batch:
                        print(f"  ℹ️  Aucun avis extrait — arrêt")
                        break

                    reviews.extend(batch)
                    print(f"  ✅ {len(batch)} avis extraits (page {page_num})")
                    page.wait_for_timeout(random.randint(2_000, 3_500))

            finally:
                browser.close()

        print(f"  🎯 Total : {len(reviews)} avis TripAdvisor")
        return reviews if reviews else self._fallback(url)

    def _page_url(self, base_url: str, page_num: int) -> str:
        """TripAdvisor pagine via 'or<N>' dans l'URL. Page 1 = pas de modificateur."""
        if page_num == 1:
            return base_url
        offset = (page_num - 1) * 10
        # Insérer '-or{offset}-' avant 'Reviews-' ou en fin d'URL
        if "Reviews-" in base_url:
            return base_url.replace("Reviews-", f"Reviews-or{offset}-", 1)
        return base_url.rstrip("/") + f"-or{offset}"

    def _close_popups(self, page):
        """Ferme les éventuelles bandeaux cookie / inscription."""
        for sel in [
            "button[id*='close']",
            "button[aria-label*='Fermer']",
            "button[aria-label*='Close']",
            "[data-automation='closeModal']",
            "[class*='closeSvg']",
        ]:
            try:
                btn = page.locator(sel).first
                if btn.is_visible(timeout=1_500):
                    btn.click()
                    page.wait_for_timeout(500)
            except Exception:
                pass

    def _expand_reviews(self, page):
        """Clique sur 'Plus' / 'Voir plus' pour déplier les avis tronqués."""
        for sel in [
            "button[data-automation='expandReview']",
            "button[class*='read-more']",
            "span[class*='readMore']",
            "//button[contains(., 'Plus')]",
            "//button[contains(., 'More')]",
        ]:
            try:
                if sel.startswith("//"):
                    btns = page.locator(f"xpath={sel}").all()
                else:
                    btns = page.locator(sel).all()
                for btn in btns[:20]:
                    try:
                        btn.click()
                        page.wait_for_timeout(200)
                    except Exception:
                        pass
                if btns:
                    print(f"  📖 {len(btns)} avis dépliés")
                    break
            except Exception:
                continue

    def _extract_reviews(self, soup: BeautifulSoup, url: str) -> List[ScrapedReview]:
        # Trouver les cartes
        cards = []
        for sel in self.CARD_SELECTORS:
            c = [x for x in soup.select(sel) if len(x.get_text(strip=True)) > 15]
            if c:
                print(f"  🔍 Sélecteur carte : '{sel}' ({len(c)})")
                cards = c
                break

        if not cards:
            print("  ⚠️  Aucune carte TripAdvisor dans le HTML")
            return []

        results = []
        for card in cards:
            # Texte
            text = None
            for sel in self.TEXT_SELECTORS:
                el = card.select_one(sel)
                if el:
                    t = el.get_text(strip=True)
                    if len(t) > 10:
                        text = t
                        break
            if not text:
                continue

            # Note — bubble_XX → XX/5
            rating = None
            bub = card.select_one("[class*='bubble_']")
            if bub:
                cls = " ".join(bub.get("class", []))
                m   = re.search(r"bubble_(\d+)", cls)
                if m:
                    v = float(m.group(1))
                    rating = v / 10 if v > 5 else v
            if rating is None:
                # Chercher svg title ou aria-label
                aria_el = card.select_one("[aria-label*='sur 5'], [aria-label*='out of 5']")
                if aria_el:
                    m = re.search(r"(\d+(?:[.,]\d+)?)", aria_el.get("aria-label", ""))
                    if m:
                        rating = float(m.group(1).replace(",", "."))

            # Date
            date = None
            d_el = card.select_one("time, [class*='date']")
            if d_el:
                date = d_el.get("datetime", d_el.get_text(strip=True))

            results.append(ScrapedReview(
                text=text, rating=rating,
                source="TripAdvisor", target=url,
                date=date, langue="fr",
            ))

        return results

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Fallback TripAdvisor (données démo)")
        data = [
            ("Cucina Napoli est un vrai coup de cœur ! Les pizzas sont authentiques et la pâte est parfaite.", 5.0),
            ("Excellent restaurant italien à Casablanca. Service rapide et accueil chaleureux.", 5.0),
            ("Bonne cuisine mais un peu cher pour Casablanca. Ambiance sympa cependant.", 4.0),
            ("Service correct, cuisine satisfaisante. Rien d'exceptionnel pour le prix.", 3.0),
            ("Déçu par la qualité des plats. Pour ce prix j'attendais mieux.", 2.0),
            ("Meziana bzzaf had la pizza ! Kan mstahel, ghir chi chwiya 3al prix.", 5.0),
            ("Très bon rapport qualité-prix. Je recommande la pizza Napolitaine.", 4.0),
            ("L'ambiance est agréable mais le service était lent lors de ma visite.", 3.0),
        ]
        return [ScrapedReview(t, r, "TripAdvisor (Démo)", url, langue="fr") for t, r in data]


# ─────────────────────────────────────────────────────────────────────────────
# Interface publique — utilisée par app/main.py
# ─────────────────────────────────────────────────────────────────────────────

def scrape_jumia(url: str, max_pages: int = 3) -> List[Dict]:
    """Scrape les avis Jumia Maroc."""
    return [r.to_dict() for r in JumiaScraper().scrape(url, max_pages)]

def scrape_gmaps(url: str, target_reviews: int = 20) -> List[Dict]:
    """Scrape les avis Google Maps."""
    return [r.to_dict() for r in GoogleMapsScraper().scrape(url, target_reviews)]

def scrape_trustpilot(url: str, max_pages: int = 3) -> List[Dict]:
    """Scrape les avis Trustpilot."""
    return [r.to_dict() for r in TrustpilotScraper().scrape(url, max_pages)]

def scrape_tripadvisor(url: str, max_pages: int = 3) -> List[Dict]:
    """Scrape les avis TripAdvisor."""
    return [r.to_dict() for r in TripAdvisorScraper().scrape(url, max_pages)]