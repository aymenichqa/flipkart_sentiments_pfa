# app/scraper.py — Moteur de scraping V3 (Cibles Marocaines)
# ══════════════════════════════════════════════════════════════════════════════
#
# Cibles :
#   JumiaScraper      → jumia.ma        (pagination classique)
#   MarjaneScraper    → marjanemall.ma  (SPA React, networkidle)
#   GoogleMapsScraper → google.com/maps (infinite scroll sur div .m6QErb)
#
# Technologie : sync_playwright (et non async) pour éviter le conflit
# d'event loop avec FastAPI qui possède déjà sa propre boucle asyncio.
#
# Prérequis :
#   pip install playwright beautifulsoup4 lxml
#   playwright install chromium
# ══════════════════════════════════════════════════════════════════════════════

import re
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# ─────────────────────────────────────────────────────────────────────────────
# Structure de données commune
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScrapedReview:
    """Représente un avis extrait depuis n'importe quelle source."""
    text   : str
    rating : Optional[float]       # Note sur 5
    source : str                   # "Jumia" | "Marjane" | "Google Maps"
    target : str                   # URL ou nom du lieu
    date   : Optional[str] = None
    langue : Optional[str] = None  # "fr" | "ar" | "darija" | "en"

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
# Classe de base — configuration Playwright partagée
# ─────────────────────────────────────────────────────────────────────────────

class BasePlaywrightScraper:
    """
    Classe de base pour tous les scrapers Playwright.
    Fournit une méthode _get_page_html() commune qui :
      - Lance Chromium en mode headless
      - Masque la signature "webdriver" (anti-détection)
      - Applique des paramètres réalistes (locale, viewport, user-agent)
    """

    # User-agent Chrome réel — change régulièrement si bloqué
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def _build_context_args(self, locale: str = "fr-MA") -> dict:
        """Arguments communs pour browser.new_context()."""
        return dict(
            user_agent  = self.USER_AGENT,
            viewport    = {"width": 1366, "height": 768},
            locale      = locale,
            timezone_id = "Africa/Casablanca",
            # Simuler un vrai navigateur marocain
            extra_http_headers = {
                "Accept-Language": "fr-MA,fr;q=0.9,ar;q=0.8,en;q=0.7",
            },
        )

    def _get_page_html(
        self,
        url         : str,
        wait_until  : str           = "domcontentloaded",
        wait_selector: Optional[str]= None,
        timeout_ms  : int           = 30_000,
        extra_wait_ms: int          = 2_000,
        locale      : str           = "fr-MA",
        actions     = None,          # callable(page) pour actions custom (scroll, etc.)
    ) -> Optional[BeautifulSoup]:
        """
        Charge une URL avec Playwright et retourne un BeautifulSoup.

        Paramètres :
            wait_until   : stratégie d'attente ("domcontentloaded", "networkidle", "load")
            wait_selector: sélecteur CSS à attendre avant de lire le HTML
            extra_wait_ms: pause supplémentaire après chargement (SPA lentes)
            actions      : fonction callable(page) pour actions personnalisées (scroll...)
        """
        print(f"  🌐 [{self.__class__.__name__}] Playwright → {url}")
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
                context = browser.new_context(**self._build_context_args(locale))
                page    = context.new_page()

                # ── Masquage webdriver (astuce anti-bot clé) ──────────────────
                page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                )

                # ── Chargement de la page ─────────────────────────────────────
                page.goto(url, wait_until=wait_until, timeout=timeout_ms)

                # ── Attendre un sélecteur CSS précis ─────────────────────────
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=10_000)
                    except PWTimeout:
                        print(f"  ⚠️  Sélecteur '{wait_selector}' non trouvé — page peut-être vide")

                # ── Actions personnalisées (scroll, click, etc.) ──────────────
                if actions:
                    actions(page)

                # ── Pause finale (SPA qui charge en arrière-plan) ─────────────
                if extra_wait_ms > 0:
                    page.wait_for_timeout(extra_wait_ms)

                html = page.content()
                browser.close()

            return BeautifulSoup(html, "lxml")

        except PWTimeout:
            print(f"  ⏰ Timeout Playwright sur : {url}")
        except Exception as e:
            print(f"  ❌ Erreur Playwright : {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# JumiaScraper — jumia.ma
# ─────────────────────────────────────────────────────────────────────────────
#
# STRATÉGIE :
# Jumia Maroc utilise une pagination classique (page=1, page=2...).
# Les avis produit sont dans des <article class="article"> ou dans
# la section "#reviews". On attend le sélecteur des articles de reviews
# avant d'extraire le HTML.
#
# URL type : https://www.jumia.ma/catalog/?q=casque+bluetooth#catalog-listing
# ou page produit : https://www.jumia.ma/samsung-galaxy-a15/...html

class JumiaScraper(BasePlaywrightScraper):
    """
    Scraper pour Jumia Maroc (jumia.ma).
    Extrait les avis clients et les notes depuis les pages produits.
    """

    BASE_URL = "https://www.jumia.ma"

    # Sélecteurs en cascade — robustesse face aux mises à jour DOM
    REVIEW_SELECTORS = [
        "article.article",           # sélecteur principal
        "[class*='revs'] article",   # variante section reviews
        "div[class*='rv-bd']",
        "[data-test='review-item']",
        ".review-item",
    ]
    TEXT_SELECTORS   = [
        "p[class*='bd']",
        ".content",
        "p",
    ]
    RATING_SELECTORS = [
        "div[class*='stars-ct']",
        "[class*='rvw-stars']",
        "[class*='stars']",
    ]

    def scrape(self, url: str, max_pages: int = 3) -> List[ScrapedReview]:
        """
        Scrape les avis d'une page produit Jumia.

        Args:
            url       : URL de la page produit (ex: https://www.jumia.ma/Samsung.../mp/)
                        OU URL de la section reviews (finit souvent par /reviews/)
            max_pages : nombre de pages de reviews à parcourir
        """
        print(f"\n🛍️  JumiaScraper — {url}")
        reviews = []

        # Jumia expose les reviews sur une URL dédiée
        # Ex: https://www.jumia.ma/<produit>/mpXXXXX/reviews/
        reviews_url = url.rstrip("/") + "/reviews/"
        if "reviews" in url:
            reviews_url = url

        for page_num in range(1, max_pages + 1):
            page_url = f"{reviews_url}?page={page_num}#reviews"
            print(f"  📄 Page {page_num}/{max_pages}")

            soup = self._get_page_html(
                url          = page_url,
                wait_until   = "domcontentloaded",
                wait_selector= "article.article, [class*='rv-bd'], #reviews",
                extra_wait_ms= 1500,
            )

            if soup is None:
                print(f"  ❌ Échec page {page_num}")
                break

            cards = self._find_review_cards(soup)
            if not cards:
                print(f"  ℹ️  Aucun avis sur la page {page_num} — fin de pagination")
                break

            n = 0
            for card in cards:
                review = self._extract_from_card(card, url)
                if review:
                    reviews.append(review)
                    n += 1

            print(f"  ✅ {n} avis extraits — page {page_num}")
            time.sleep(random.uniform(1.5, 3.0))  # politesse serveur

        print(f"  🎯 Total : {len(reviews)} avis Jumia")
        return reviews if reviews else self._fallback(url)

    def _find_review_cards(self, soup: BeautifulSoup):
        for sel in self.REVIEW_SELECTORS:
            cards = soup.select(sel)
            if cards:
                print(f"  🔍 Sélecteur actif : '{sel}' ({len(cards)} blocs)")
                return cards
        return []

    def _extract_from_card(self, card: BeautifulSoup, url: str) -> Optional[ScrapedReview]:
        # Texte de l'avis
        text = None
        for sel in self.TEXT_SELECTORS:
            el = card.select_one(sel)
            if el:
                t = el.get_text(strip=True)
                if t and len(t) > 8:
                    text = t
                    break

        if not text:
            # Fallback : texte brut de la carte
            for tag in card.find_all(["button", "a", "img", "svg", "span[class*='icon']"]):
                tag.decompose()
            text = card.get_text(separator=" ", strip=True)
            if len(text) < 8:
                return None

        # Note (étoiles)
        rating = None
        for sel in self.RATING_SELECTORS:
            el = card.select_one(sel)
            if el:
                # Jumia encode souvent la note dans un attribut style ou class
                cls = " ".join(el.get("class", []))
                m   = re.search(r"(\d+(?:\.\d+)?)", el.get_text() or cls)
                if m:
                    rating = float(m.group(1))
                    if rating > 5:       # parfois encodé sur 50 ou 100
                        rating = rating / 10
                    break

        # Date
        date = None
        date_el = card.select_one("time, [class*='date']")
        if date_el:
            date = date_el.get("datetime", date_el.get_text(strip=True))

        return ScrapedReview(
            text=text, rating=rating,
            source="Jumia Maroc", target=url,
            date=date, langue="fr",
        )

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Données démo Jumia (fallback)")
        data = [
            ("Produit reçu rapidement, qualité conforme à la description. Je recommande !", 5.0),
            ("Bonne qualité pour le prix. Livraison en 3 jours. Satisfait.", 4.0),
            ("Produit correct, rien d'exceptionnel. L'emballage était abîmé.", 3.0),
            ("Pas terrible du tout, la qualité est décevante par rapport aux photos.", 2.0),
            ("Arnaque ! Produit reçu complètement différent de ce qui était affiché.", 1.0),
            ("Très bon rapport qualité-prix. Jumia livraison impeccable comme toujours.", 5.0),
            ("Mazyan bzzaf, ghir chi mochkila m3a l packaging kayn.", 4.0),  # darija
        ]
        return [
            ScrapedReview(text=t, rating=r, source="Jumia (Démo)", target=url, langue="fr")
            for t, r in data
        ]


# ─────────────────────────────────────────────────────────────────────────────
# MarjaneScraper — marjanemall.ma
# ─────────────────────────────────────────────────────────────────────────────
#
# STRATÉGIE :
# Marjane Mall est une SPA (Single Page Application) React.
# Les avis sont chargés dynamiquement via des appels XHR.
# On doit attendre "networkidle" (plus d'activité réseau) avant d'extraire.
# Les sélecteurs sont souvent des classes générées (ex: "css-1abc23").
# On utilise des data-* attributes pour être plus robuste.

class MarjaneScraper(BasePlaywrightScraper):
    """
    Scraper pour Marjane Mall (marjanemall.ma).
    Utilise networkidle pour attendre le rendu complet de la SPA React.
    """

    BASE_URL = "https://marjanemall.ma"

    REVIEW_SELECTORS = [
        "[class*='review']",
        "[class*='Review']",
        "[class*='comment']",
        "[class*='avis']",
        "[data-testid*='review']",
        ".chakra-stack",             # Marjane utilise Chakra UI
    ]
    TEXT_SELECTORS = [
        "p[class*='body']",
        "[class*='text']",
        "[class*='content']",
        "p",
    ]
    RATING_SELECTORS = [
        "[class*='rating']",
        "[class*='star']",
        "[aria-label*='étoile']",
        "[aria-label*='star']",
        "[aria-label*='out of']",
    ]

    def scrape(self, url: str, max_pages: int = 2) -> List[ScrapedReview]:
        """
        Scrape les avis d'une page produit Marjane Mall.

        Args:
            url : URL de la page produit Marjane
                  ex: https://marjanemall.ma/products/smartphone-xyz
        """
        print(f"\n🏪 MarjaneScraper — {url}")
        reviews = []

        for page_num in range(1, max_pages + 1):
            # Marjane utilise parfois un paramètre "page" ou "avis_page"
            page_url = url if page_num == 1 else f"{url}?page={page_num}"
            print(f"  📄 Page {page_num}/{max_pages}")

            def actions_marjane(page):
                """Actions après chargement : scroll pour déclencher lazy-load."""
                try:
                    # Scroller vers la section des avis
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    page.wait_for_timeout(1500)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1500)
                except Exception:
                    pass

            soup = self._get_page_html(
                url          = page_url,
                # networkidle = attendre que tous les appels XHR soient terminés
                wait_until   = "networkidle",
                wait_selector= ", ".join(self.REVIEW_SELECTORS[:3]),
                timeout_ms   = 45_000,   # SPAs peuvent être lentes
                extra_wait_ms= 2500,
                actions      = actions_marjane,
            )

            if soup is None:
                break

            cards = self._find_review_cards(soup)
            if not cards:
                print(f"  ℹ️  Aucun avis page {page_num}")
                break

            n = 0
            for card in cards:
                review = self._extract_from_card(card, url)
                if review:
                    reviews.append(review)
                    n += 1

            print(f"  ✅ {n} avis extraits")
            time.sleep(random.uniform(2.0, 3.5))

        print(f"  🎯 Total : {len(reviews)} avis Marjane")
        return reviews if reviews else self._fallback(url)

    def _find_review_cards(self, soup: BeautifulSoup):
        for sel in self.REVIEW_SELECTORS:
            cards = soup.select(sel)
            # Filtrer les cartes trop courtes (menus, boutons...)
            valid = [c for c in cards if len(c.get_text(strip=True)) > 20]
            if valid:
                print(f"  🔍 Sélecteur : '{sel}' ({len(valid)} blocs valides)")
                return valid
        return []

    def _extract_from_card(self, card: BeautifulSoup, url: str) -> Optional[ScrapedReview]:
        text = None
        for sel in self.TEXT_SELECTORS:
            el = card.select_one(sel)
            if el:
                t = el.get_text(strip=True)
                if t and len(t) > 10:
                    text = t
                    break

        if not text:
            for tag in card.find_all(["button", "svg", "img"]):
                tag.decompose()
            text = card.get_text(separator=" ", strip=True)
            if len(text) < 10:
                return None

        rating = None
        for sel in self.RATING_SELECTORS:
            el = card.select_one(sel)
            if el:
                aria = el.get("aria-label", "")
                m    = re.search(r"(\d+(?:\.\d+)?)", aria or el.get_text())
                if m:
                    v = float(m.group(1))
                    rating = v if v <= 5 else v / 10
                    break
            # Compter les étoiles pleines
            stars = card.select("[class*='filled'], [class*='active'], [color='orange']")
            if stars:
                rating = float(len(stars))
                break

        return ScrapedReview(
            text=text, rating=rating,
            source="Marjane Mall", target=url,
            langue="fr",
        )

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Données démo Marjane (fallback)")
        data = [
            ("Excellent produit, correspond parfaitement à la description. Marjane toujours fiable.", 5.0),
            ("Bonne qualité, livraison rapide. Je recommande ce vendeur.", 4.0),
            ("Produit acceptable pour le prix. Pas de problème particulier.", 3.0),
            ("Déçu par la qualité. Pour ce prix je m'attendais à mieux.", 2.0),
            ("Produit de mauvaise qualité. Retour en cours.", 1.0),
            ("Super qualité, hada produit mezian bzzaf, kan mstahel.", 5.0),  # darija
            ("Correct mais les délais de livraison sont un peu longs.", 3.0),
        ]
        return [
            ScrapedReview(text=t, rating=r, source="Marjane (Démo)", target=url, langue="fr")
            for t, r in data
        ]


# ─────────────────────────────────────────────────────────────────────────────
# GoogleMapsScraper — Avis de commerces locaux
# ─────────────────────────────────────────────────────────────────────────────
#
# STRATÉGIE :
# Google Maps charge les avis via infinite scroll dans la div ".m6QErb".
# Il faut :
#   1. Ouvrir la page Google Maps du commerce
#   2. Cliquer sur l'onglet "Avis"
#   3. Scroller DANS la div .m6QErb (pas window.scrollTo — ça ne marche pas)
#   4. Répéter le scroll jusqu'à avoir assez d'avis
#   5. Extraire le HTML et parser avec BeautifulSoup

class GoogleMapsScraper(BasePlaywrightScraper):
    """
    Scraper Google Maps pour les avis de commerces locaux marocains.
    Gère l'infinite scroll de la div .m6QErb.

    Argument `url` :
        URL directe Google Maps du commerce
        ex: https://www.google.com/maps/place/Marjane+Casa+Anfa/@33.57...,7z/data=...
        OU URL courte : https://maps.google.com/?cid=XXXXXXXXXX
    """

    # Sélecteur de la div scrollable contenant les avis
    REVIEWS_DIV_SELECTOR = "div.m6QErb"

    # Sélecteurs pour les cartes d'avis
    REVIEW_CARD_SELECTOR = "div[data-review-id], [class*='jftiEf'], [class*='GHT2ce']"

    # Sélecteurs texte/note
    TEXT_SELECTOR   = "[class*='wiI7pd'], span[class*='HPa7od'], [class*='review-full-text']"
    RATING_SELECTOR = "span[role='img'][aria-label*='étoile'], span[role='img'][aria-label*='star']"
    DATE_SELECTOR   = "span[class*='rsqaWe']"

    def scrape(self, url: str, target_reviews: int = 20) -> List[ScrapedReview]:
        """
        Scrape les avis Google Maps d'un commerce.

        Args:
            url            : URL Google Maps du commerce
            target_reviews : nombre d'avis cibles à collecter (scroll jusqu'à ce nombre)
        """
        print(f"\n📍 GoogleMapsScraper — {url}")
        print(f"  🎯 Objectif : {target_reviews} avis")

        def actions_gmaps(page):
            """
            Actions clés pour Google Maps :
            1. Attendre le chargement complet
            2. Cliquer sur l'onglet "Avis"
            3. Scroller DANS la div .m6QErb (pas window.scrollTo)
            """
            # Attendre que la page soit bien chargée
            page.wait_for_timeout(3000)

            # Essayer de cliquer sur l'onglet "Avis" / "Reviews"
            tab_selectors = [
                "button[aria-label*='Avis']",
                "button[aria-label*='Reviews']",
                "[data-tab-index='1']",
                "//button[contains(., 'Avis')]",
                "//button[contains(., 'Reviews')]",
            ]
            for sel in tab_selectors:
                try:
                    if sel.startswith("//"):
                        tab = page.locator(f"xpath={sel}").first
                    else:
                        tab = page.locator(sel).first
                    tab.click(timeout=3000)
                    print("  ✅ Onglet Avis cliqué")
                    page.wait_for_timeout(2500)
                    break
                except Exception:
                    continue

            # Attendre que la div scrollable apparaisse
            try:
                page.wait_for_selector(self.REVIEWS_DIV_SELECTOR, timeout=8000)
            except PWTimeout:
                print("  ⚠️  Div .m6QErb non trouvée — peut-être pas d'onglet avis")
                return

            # ── Infinite scroll dans la div .m6QErb ──────────────────────────
            # IMPORTANT : il faut scroller la div elle-même, pas window
            scroll_script = """
                (selector) => {
                    const div = document.querySelector(selector);
                    if (div) {
                        div.scrollTop += 800;
                        return div.scrollTop;
                    }
                    return 0;
                }
            """

            current_count = 0
            max_scrolls   = target_reviews // 5 + 8   # ~5 avis par scroll, +marge
            no_new_count  = 0

            for i in range(max_scrolls):
                # Compter les avis actuellement chargés
                cards = page.query_selector_all(self.REVIEW_CARD_SELECTOR)
                new_count = len(cards)

                if new_count >= target_reviews:
                    print(f"  🎯 {new_count} avis chargés — objectif atteint")
                    break

                if new_count == current_count:
                    no_new_count += 1
                    if no_new_count >= 4:
                        print(f"  ℹ️  Plus d'avis à charger ({new_count} au total)")
                        break
                else:
                    no_new_count = 0
                    print(f"  📜 Scroll {i+1} — {new_count} avis chargés...")

                current_count = new_count

                # Scroller dans la div
                page.evaluate(scroll_script, self.REVIEWS_DIV_SELECTOR)
                page.wait_for_timeout(random.randint(1200, 2200))

            # Déplier les avis tronqués ("Voir plus")
            expand_selectors = [
                "button[aria-label*='Voir plus']",
                "button[aria-label*='See more']",
                ".w8nwRe",
                "[jsaction*='review-full-text']",
            ]
            for sel in expand_selectors:
                try:
                    btns = page.query_selector_all(sel)
                    for btn in btns[:15]:  # max 15 boutons "Voir plus"
                        try:
                            btn.click()
                            page.wait_for_timeout(300)
                        except Exception:
                            pass
                    if btns:
                        print(f"  📖 {len(btns)} avis dépliés ('Voir plus')")
                    break
                except Exception:
                    continue

        soup = self._get_page_html(
            url          = url,
            wait_until   = "domcontentloaded",
            wait_selector= self.REVIEWS_DIV_SELECTOR,
            timeout_ms   = 45_000,
            extra_wait_ms= 1000,
            locale       = "fr-MA",
            actions      = actions_gmaps,
        )

        if soup is None:
            print("  ❌ Échec du chargement — fallback démo")
            return self._fallback(url)

        reviews = self._extract_reviews(soup, url)
        print(f"  🎯 Total : {len(reviews)} avis Google Maps")
        return reviews if reviews else self._fallback(url)

    def _extract_reviews(self, soup: BeautifulSoup, url: str) -> List[ScrapedReview]:
        reviews = []

        # Sélecteurs de cartes en cascade
        card_selectors = [
            "div[data-review-id]",
            "[class*='jftiEf']",
            "[class*='GHT2ce']",
            ".review-item",
        ]
        cards = []
        for sel in card_selectors:
            cards = soup.select(sel)
            if cards:
                print(f"  🔍 Sélecteur : '{sel}' ({len(cards)} avis)")
                break

        if not cards:
            print("  ⚠️  Aucune carte d'avis trouvée dans le HTML")
            return []

        for card in cards:
            # ── Texte ────────────────────────────────────────────────────────
            text = None
            for sel in [
                "[class*='wiI7pd']",
                "span[class*='HPa7od']",
                "[data-expanded-text]",
                ".MyEned span",
                "span",
            ]:
                el = card.select_one(sel)
                if el:
                    t = el.get_text(strip=True)
                    if t and len(t) > 5:
                        text = t
                        break

            if not text:
                continue

            # ── Note ─────────────────────────────────────────────────────────
            rating = None
            r_el   = card.select_one("span[role='img'][aria-label]")
            if r_el:
                aria = r_el.get("aria-label", "")
                # Format: "4 étoiles" ou "4 stars" ou "Rated 4.0 out of 5"
                m = re.search(r"(\d+(?:[.,]\d+)?)", aria)
                if m:
                    rating = float(m.group(1).replace(",", "."))

            # ── Date ─────────────────────────────────────────────────────────
            date = None
            d_el = card.select_one("[class*='rsqaWe']")
            if d_el:
                date = d_el.get_text(strip=True)

            reviews.append(ScrapedReview(
                text=text, rating=rating,
                source="Google Maps", target=url,
                date=date, langue="fr",
            ))

        return reviews

    def _fallback(self, url: str) -> List[ScrapedReview]:
        print("  📦 Données démo Google Maps (fallback)")
        data = [
            ("Très bon service, personnel accueillant. Je reviendrai sans hésiter !", 5.0),
            ("Super endroit, propre et bien organisé. Les prix sont raisonnables.", 5.0),
            ("Service correct, temps d'attente acceptable. Rien d'exceptionnel.", 3.0),
            ("Déçu par le service. Longue attente et personnel peu aimable.", 2.0),
            ("Endroit bien. Baghi ykon hadshi hsen, walakin mezian 3ala had taman.", 4.0),  # darija
            ("Très bonne expérience. Je recommande vivement à tous.", 5.0),
            ("Moyen. Kan mchkil m3a l'parking w les prix 3alin chwiya.", 3.0),  # darija
        ]
        return [
            ScrapedReview(text=t, rating=r, source="Google Maps (Démo)", target=url, langue="fr")
            for t, r in data
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires — interface publique pour app/main.py
# ─────────────────────────────────────────────────────────────────────────────

def scrape_jumia(url: str, max_pages: int = 3) -> List[Dict]:
    """Scrape les avis Jumia Maroc. Retourne une liste de dicts."""
    return [r.to_dict() for r in JumiaScraper().scrape(url, max_pages)]

def scrape_marjane(url: str, max_pages: int = 2) -> List[Dict]:
    """Scrape les avis Marjane Mall. Retourne une liste de dicts."""
    return [r.to_dict() for r in MarjaneScraper().scrape(url, max_pages)]

def scrape_gmaps(url: str, target_reviews: int = 20) -> List[Dict]:
    """Scrape les avis Google Maps. Retourne une liste de dicts."""
    return [r.to_dict() for r in GoogleMapsScraper().scrape(url, target_reviews)]