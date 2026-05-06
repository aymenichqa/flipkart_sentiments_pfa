# app/scraper.py — Module de scraping robuste
# TrustpilotScraper utilise sync_playwright (Chromium) pour bypasser Cloudflare WAF
# AmazonScraper utilise sync_playwright pour éviter les CAPTCHAs Amazon
#
# Prérequis :
#   pip install playwright
#   playwright install chromium

import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time
import random
import re
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Structure de données commune
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScrapedReview:
    """Structure standardisée pour une review scrapée."""
    text: str
    rating: Optional[float]
    source: str
    target: str
    date: Optional[str]
    url: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Classe de base
# ─────────────────────────────────────────────────────────────────────────────

class BaseScraper:
    """Classe de base avec session requests et délai anti-bot."""

    def __init__(self, delay_range: tuple = (1.5, 3.0)):
        self.delay_min, self.delay_max = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        })

    def _random_delay(self):
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    def _safe_request(self, url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
        """Requête HTTP standard (requests)."""
        try:
            self._random_delay()
            response = self.session.get(url, timeout=timeout)
            print(f"  📡 {response.status_code} — {url}")
            response.raise_for_status()
            return BeautifulSoup(response.text, 'lxml')
        except requests.exceptions.Timeout:
            print(f"  ⏰ Timeout : {url}")
        except requests.exceptions.HTTPError as e:
            print(f"  🚫 HTTP {e.response.status_code} : {url}")
        except Exception as e:
            print(f"  ❌ Erreur : {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TrustpilotScraper — sync_playwright (bypass Cloudflare WAF)
# ─────────────────────────────────────────────────────────────────────────────

class TrustpilotScraper(BaseScraper):
    """Scraper Trustpilot utilisant Playwright (synchrone) pour bypasser Cloudflare."""

    BASE_URL = "https://www.trustpilot.com"

    def _safe_request(self, url: str) -> Optional[BeautifulSoup]:
        print(f"  🌐 Playwright (sync) → {url}")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled",
                          "--no-sandbox"]
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 800},
                    locale="en-US",
                )
                page = context.new_page()
                page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                )

                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3500)

                html_content = page.content()
                browser.close()

                if "cf_chl_opt" in html_content or "Just a moment" in html_content:
                    print("  🚫 Cloudflare challenge détecté — Playwright bloqué")
                    return None

                return BeautifulSoup(html_content, 'lxml')

        except Exception as e:
            print(f"  ❌ Playwright erreur : {e}")
            return None

    def scrape(self, company: str, max_pages: int = 3) -> List[ScrapedReview]:
        reviews = []
        for page_num in range(1, max_pages + 1):
            url = f"{self.BASE_URL}/review/{company}?page={page_num}&languages=en"
            print(f"  📄 Page {page_num}/{max_pages}")
            soup = self._safe_request(url)

            if soup is None:
                print(f"  ❌ Échec page {page_num} → arrêt")
                break

            cards = self._find_review_cards(soup)

            if not cards:
                print(f"  ⚠️  Aucun bloc review trouvé page {page_num}")
                if self._is_error_page(soup):
                    print(f"  🚫 Entreprise '{company}' introuvable sur Trustpilot")
                break

            page_count = 0
            for card in cards:
                review = self._extract_review(card, company, url)
                if review:
                    reviews.append(review)
                    page_count += 1

            print(f"  ✅ {page_count} reviews — page {page_num}")

            if page_count == 0:
                break

            time.sleep(random.uniform(2.0, 4.0))

        print(f"  🎯 Total : {len(reviews)} reviews pour '{company}'")

        if not reviews:
            print("  📦 0 reviews réelles → données démo")
            return self._get_fallback_reviews(company)

        return reviews

    def _find_review_cards(self, soup: BeautifulSoup) -> List[BeautifulSoup]:
        selectors = [
            'article[data-service-review-card-paper]',
            '[class*="styles_reviewCardInner"]',
            '[data-review-id]',
            '.review-card',
            '[class*="review"]',
        ]
        for sel in selectors:
            cards = soup.select(sel)
            if cards:
                print(f"  🔍 Sélecteur : {sel} ({len(cards)} blocs)")
                return cards
        return []

    def _extract_review(self, card: BeautifulSoup, company: str, url: str) -> Optional[ScrapedReview]:
        try:
            text = self._extract_text(card)
            if not text or len(text.strip()) < 10:
                return None
            return ScrapedReview(
                text=text.strip(),
                rating=self._extract_rating(card),
                source="Trustpilot",
                target=company,
                date=self._extract_date(card),
                url=url
            )
        except Exception as e:
            print(f"  ⚠️  Extraction erreur : {e}")
            return None

    def _extract_text(self, card: BeautifulSoup) -> Optional[str]:
        for sel in [
            'p[class*="typography_body"]',
            '[data-service-review-text-typography]',
            '[class*="review-content"] p',
            '[class*="review-text"]',
            'p',
        ]:
            el = card.select_one(sel)
            if el:
                t = el.get_text(strip=True)
                if t and len(t) > 10:
                    return t
        return None

    def _extract_rating(self, card: BeautifulSoup) -> Optional[float]:
        el = card.select_one('[data-service-review-rating]')
        if el:
            try:
                return float(el.get('data-service-review-rating', 0))
            except:
                pass
        stars = card.select_one('[class*="star-rating"]')
        if stars:
            m = re.search(r'(\d+(?:\.\d+)?)', stars.get_text())
            if m:
                return float(m.group(1))
        return None

    def _extract_date(self, card: BeautifulSoup) -> Optional[str]:
        el = card.select_one('time')
        if el and el.get('datetime'):
            return el['datetime'][:10]
        return None

    def _is_error_page(self, soup: BeautifulSoup) -> bool:
        txt = soup.get_text().lower()
        return any(x in txt for x in ["company not found", "page not found", "404"])

    def _get_fallback_reviews(self, company: str) -> List[ScrapedReview]:
        print(f"  📦 Données démo pour '{company}'")
        demo = [
            ("Great product, exceeded my expectations! Fast delivery and excellent customer service.", 5.0),
            ("Very good quality and reasonable price. Would definitely recommend to friends.", 5.0),
            ("Amazing experience. Product arrived on time and was exactly as described.", 5.0),
            ("Good value for money, though packaging could be better.", 4.0),
            ("Average product. It works but nothing special. Support was helpful though.", 3.0),
            ("Disappointed with the quality. Product arrived damaged. Difficult return process.", 2.0),
            ("Worst purchase ever. Complete waste of money. Do not recommend.", 1.0),
        ]
        dates = ["2026-05-05","2026-05-04","2026-05-03","2026-05-02","2026-05-01","2026-04-30","2026-04-29"]
        return [
            ScrapedReview(text=t, rating=r, source="Trustpilot (Demo)",
                          target=company, date=d)
            for (t, r), d in zip(demo, dates)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# AmazonScraper — Mis à jour avec Playwright pour éviter les CAPTCHAs
# ─────────────────────────────────────────────────────────────────────────────

class AmazonScraper(BaseScraper):
    """Scraper Amazon.in — sélecteurs data-hook stables + Playwright."""

    BASE_URL = "https://www.amazon.in"

    def _safe_request_playwright(self, url: str) -> Optional[BeautifulSoup]:
        """Utilise Playwright pour charger la page et éviter les blocages Amazon."""
        print(f"  🌐 Playwright Amazon → {url}")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    locale="en-IN"
                )
                page = context.new_page()
                page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3500) # Pause pour laisser les reviews charger
                
                html_content = page.content()
                browser.close()

                # Vérification si Amazon a quand même affiché un CAPTCHA
                if "Type the characters you see in this image" in html_content or "Enter the characters you see below" in html_content:
                    print("  🚫 Amazon CAPTCHA détecté.")
                    return None
                    
                return BeautifulSoup(html_content, 'lxml')
        except Exception as e:
            print(f"  ❌ Erreur Playwright Amazon : {e}")
            return None

    def scrape(self, asin: str, max_pages: int = 2) -> List[ScrapedReview]:
        reviews = []
        for page in range(1, max_pages + 1):
            url = f"{self.BASE_URL}/product-reviews/{asin}?pageNumber={page}&reviewerType=all_reviews"
            print(f"  📄 Page {page}/{max_pages}")
            
            soup = self._safe_request_playwright(url)
            
            if soup is None:
                break
                
            cards = soup.select('[data-hook="review"]')
            if not cards:
                print(f"  ⚠️  Aucune review page {page} (Sélecteur introuvable ou fin des pages)")
                break
                
            page_count = 0
            for card in cards:
                r = self._extract_review_amazon(card, asin, url)
                if r:
                    reviews.append(r)
                    page_count += 1
                    
            print(f"  ✅ {page_count} reviews — page {page}")
            
        print(f"  🎯 Total : {len(reviews)} reviews pour ASIN '{asin}'")
        
        # --- AJOUT DU FALLBACK ICI ---
        if not reviews:
            print("  📦 0 reviews récupérées sur Amazon → données démo (produits)")
            return self._get_fallback_reviews_amazon(asin)
            
        return reviews

    def _extract_review_amazon(self, card: BeautifulSoup, asin: str, url: str) -> Optional[ScrapedReview]:
        try:
            text_el = card.select_one('[data-hook="review-body"]')
            if not text_el:
                return None
            text = text_el.get_text(strip=True)
            if not text or len(text) < 10:
                return None

            rating = None
            stars_el = card.select_one('[data-hook="review-star-rating"]')
            if stars_el:
                try:
                    rating = float(stars_el.get_text().strip().split()[0])
                except:
                    pass

            date = None
            date_el = card.select_one('[data-hook="review-date"]')
            if date_el:
                date = date_el.get_text(strip=True)

            return ScrapedReview(text=text, rating=rating,
                                 source="Amazon.in", target=asin,
                                 date=date, url=url)
        except Exception as e:
            print(f"  ⚠️  Erreur extraction Amazon : {e}")
            return None

    # --- NOUVELLE METHODE FALLBACK ---
    def _get_fallback_reviews_amazon(self, asin: str) -> List[ScrapedReview]:
        print(f"  📦 Données démo générées pour l'ASIN '{asin}'")
        demo = [
            ("The battery life on this phone is amazing and the screen is very clear. Best purchase ever!", 5.0),
            ("Good smartphone, fast processor. The camera is decent but struggles in low light.", 4.0),
            ("Very bad quality. The display stopped working after two days and the camera is blurry.", 1.0),
            ("It's okay for the price. Not great, not terrible. Does the basic job.", 3.0),
            ("Absolutely love the design and the smooth interface. Highly recommended product.", 5.0),
            ("Overpriced garbage. The battery drains in 3 hours and it overheats constantly.", 1.0),
        ]
        dates = ["2026-05-05","2026-05-04","2026-05-03","2026-05-02","2026-05-01","2026-04-30"]
        return [
            ScrapedReview(text=t, rating=r, source="Amazon.in (Demo)",
                          target=asin, date=d)
            for (t, r), d in zip(demo, dates)
        ]


def scrape_trustpilot(company: str, max_pages: int = 3) -> List[Dict]:
    scraper = TrustpilotScraper()
    return [
        {"text": r.text, "rating": r.rating, "source": r.source,
         "target": r.target, "date": r.date}
        for r in scraper.scrape(company, max_pages)
    ]

def scrape_amazon(asin: str, max_pages: int = 2) -> List[Dict]:
    scraper = AmazonScraper()
    return [
        {"text": r.text, "rating": r.rating, "source": r.source,
         "target": r.target, "date": r.date}
        for r in scraper.scrape(asin, max_pages)
    ]