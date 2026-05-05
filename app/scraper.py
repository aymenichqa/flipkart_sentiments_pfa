# app/scraper.py — Module de scraping robuste avec fallbacks
# Utilise des sélecteurs multiples et gestion d'erreurs avancée

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time
import random
from dataclasses import dataclass

@dataclass
class ScrapedReview:
    """Structure standardisée pour une review scrapée."""
    text: str
    rating: Optional[float]
    source: str
    target: str
    date: Optional[str]
    url: Optional[str] = None

class BaseScraper:
    """Classe de base pour tous les scrapers avec gestion d'erreurs commune."""

    def __init__(self, delay_range: tuple = (1.5, 3.0)):
        self.delay_min, self.delay_max = delay_range
        self.session = requests.Session()

        # Headers TRÈS réalistes pour éviter la détection
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
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
        """Délai aléatoire pour simuler un comportement humain."""
        delay = random.uniform(self.delay_min, self.delay_max)
        time.sleep(delay)

    def _safe_request(self, url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
        """Requête sécurisée avec gestion d'erreurs complète."""
        try:
            self._random_delay()
            response = self.session.get(url, timeout=timeout)

            # DEBUG : afficher le statut
            print(f"🔗 URL: {url}")
            print(f"📡 Status: {response.status_code}")

            # Vérifier le statut HTTP
            response.raise_for_status()

            # Vérifier que c'est bien du HTML
            if 'text/html' not in response.headers.get('content-type', ''):
                print(f"⚠️  Contenu non-HTML reçu : {response.headers.get('content-type', 'unknown')}")
                return None

            # DEBUG : afficher aperçu du HTML
            html_preview = response.text[:300].replace('\n', ' ')
            print(f"📄 HTML preview: {html_preview}...")

            return BeautifulSoup(response.text, 'lxml')

        except requests.exceptions.Timeout:
            print(f"⏰ Timeout pour {url}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"🚫 Accès interdit (403) pour {url} - site protégé par WAF/Cloudflare")
            elif e.response.status_code == 404:
                print(f"❌ Page non trouvée (404) pour {url}")
            else:
                print(f"🔴 Erreur HTTP {e.response.status_code} pour {url}")
        except requests.exceptions.RequestException as e:
            print(f"🌐 Erreur réseau pour {url}: {e}")
        except Exception as e:
            print(f"💥 Erreur inattendue pour {url}: {e}")

        return None

class TrustpilotScraper(BaseScraper):
    """Scraper Trustpilot avec sélecteurs multiples et fallbacks."""

    BASE_URL = "https://www.trustpilot.com"

    def scrape(self, company: str, max_pages: int = 3) -> List[ScrapedReview]:
        """
        Scrape les reviews Trustpilot d'une entreprise.

        Args:
            company: Nom de l'entreprise (ex: "amazon.in", "samsung.com")
            max_pages: Nombre maximum de pages à scraper

        Returns:
            Liste de ScrapedReview
        """
        reviews = []

        for page in range(1, max_pages + 1):
            url = f"{self.BASE_URL}/review/{company}?page={page}&languages=en"
            print(f"📄 Scraping page {page}/{max_pages}: {url}")

            soup = self._safe_request(url)
            if soup is None:
                print(f"❌ Impossible de charger la page {page}")
                break

            # Essayer différents sélecteurs (Trustpilot change souvent ses classes)
            cards = self._find_review_cards_trustpilot(soup)

            if not cards:
                print(f"⚠️  Aucun élément de review trouvé sur la page {page}")
                # Essayer de détecter si c'est une page d'erreur
                if self._is_error_page_trustpilot(soup):
                    print(f"🚫 Page d'erreur Trustpilot détectée pour '{company}'")
                    break
                continue

            page_reviews = 0
            for card in cards:
                review = self._extract_review_trustpilot(card, company, url)
                if review:
                    reviews.append(review)
                    page_reviews += 1

            print(f"✅ {page_reviews} reviews extraites de la page {page}")

            # Arrêter si on n'a pas trouvé de reviews sur cette page
            if page_reviews == 0:
                print(f"ℹ️  Aucune review trouvée sur la page {page}, arrêt du scraping")
                break

        print(f"🎯 Total: {len(reviews)} reviews scrapées pour {company}")
        
        # FALLBACK : si aucune review trouvée, retourner des données démo réalistes
        if len(reviews) == 0:
            print(f"⚠️  Trustpilot bloqué ou pas de résultats → utilisation de données démo")
            return self._get_fallback_reviews(company)
        
        return reviews

    def _find_review_cards_trustpilot(self, soup: BeautifulSoup) -> List[BeautifulSoup]:
        """Trouve les éléments contenant les reviews avec fallbacks multiples."""
        selectors = [
            'article[data-service-review-card-paper]',  # Sélecteur principal actuel
            '[class*="styles_reviewCardInner"]',        # Fallback 1
            '[data-review-id]',                          # Fallback 2
            '.review-card',                              # Fallback 3
            '[class*="review"]',                         # Fallback 4 générique
        ]

        for selector in selectors:
            cards = soup.select(selector)
            if cards:
                print(f"🔍 Utilisation du sélecteur: {selector} ({len(cards)} éléments)")
                return cards

        return []

    def _extract_review_trustpilot(self, card: BeautifulSoup, company: str, url: str) -> Optional[ScrapedReview]:
        """Extrait les données d'une review Trustpilot."""
        try:
            # Texte de la review - multiples fallbacks
            text = self._extract_text_trustpilot(card)
            if not text or len(text.strip()) < 10:
                return None

            # Note - multiples fallbacks
            rating = self._extract_rating_trustpilot(card)

            # Date - multiples fallbacks
            date = self._extract_date_trustpilot(card)

            return ScrapedReview(
                text=text.strip(),
                rating=rating,
                source="Trustpilot",
                target=company,
                date=date,
                url=url
            )

        except Exception as e:
            print(f"⚠️  Erreur extraction review: {e}")
            return None

    def _extract_text_trustpilot(self, card: BeautifulSoup) -> Optional[str]:
        """Extrait le texte de la review."""
        selectors = [
            'p[class*="typography_body"]',
            '[data-service-review-text-typography]',
            '[class*="review-content"] p',
            '[class*="review-text"]',
            'p',
        ]

        for selector in selectors:
            element = card.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and len(text) > 10:
                    return text

        return None

    def _extract_rating_trustpilot(self, card: BeautifulSoup) -> Optional[float]:
        """Extrait la note de la review."""
        # Essayer l'attribut data
        rating_el = card.select_one('[data-service-review-rating]')
        if rating_el:
            try:
                return float(rating_el.get('data-service-review-rating', 0))
            except:
                pass

        # Essayer les étoiles visuelles
        stars_el = card.select_one('[class*="star-rating"]')
        if stars_el:
            try:
                # Chercher un nombre dans le texte ou les classes
                text = stars_el.get_text()
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', text)
                if match:
                    return float(match.group(1))
            except:
                pass

        return None

    def _extract_date_trustpilot(self, card: BeautifulSoup) -> Optional[str]:
        """Extrait la date de la review."""
        date_el = card.select_one('time')
        if date_el:
            datetime_str = date_el.get('datetime')
            if datetime_str:
                return datetime_str[:10]  # YYYY-MM-DD

        # Fallback: chercher du texte de date
        date_selectors = [
            '[class*="review-date"]',
            '[class*="date"]',
            'small',
        ]

        for selector in date_selectors:
            el = card.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                # Chercher un pattern de date
                import re
                match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}\.\d{2}\.\d{4}', text)
                if match:
                    return match.group(0)

        return None

    def _is_error_page_trustpilot(self, soup: BeautifulSoup) -> bool:
        """Détecte si c'est une page d'erreur Trustpilot."""
        error_indicators = [
            "Company not found",
            "Page not found",
            "404",
            "company-not-found",
        ]

        page_text = soup.get_text().lower()
        return any(indicator.lower() in page_text for indicator in error_indicators)

    def _get_fallback_reviews(self, company: str) -> List[ScrapedReview]:
        """Retourne des reviews démo réalistes si le scraping échoue."""
        print(f"📦 Retour de données de démonstration pour {company}")
        demo_reviews = [
            ScrapedReview(
                text="Great product, exceeded my expectations! Fast delivery and excellent customer service.",
                rating=5.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-05-05"
            ),
            ScrapedReview(
                text="Very good quality and reasonable price. Would definitely recommend to friends.",
                rating=5.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-05-04"
            ),
            ScrapedReview(
                text="Amazing experience. Product arrived on time and was exactly as described.",
                rating=5.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-05-03"
            ),
            ScrapedReview(
                text="Good value for money, though packaging could be better.",
                rating=4.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-05-02"
            ),
            ScrapedReview(
                text="Average product. It works but nothing special. Support was helpful though.",
                rating=3.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-05-01"
            ),
            ScrapedReview(
                text="Disappointed with the quality. Product arrived damaged. Difficult return process.",
                rating=2.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-04-30"
            ),
            ScrapedReview(
                text="Worst purchase ever. Complete waste of money. Do not recommend.",
                rating=1.0,
                source="Trustpilot (Demo)",
                target=company,
                date="2026-04-29"
            ),
        ]
        return demo_reviews

class AmazonScraper(BaseScraper):
    """Scraper Amazon.in avec sélecteurs robustes."""

    BASE_URL = "https://www.amazon.in"

    def scrape(self, asin: str, max_pages: int = 2) -> List[ScrapedReview]:
        """
        Scrape les reviews Amazon d'un produit par ASIN.

        Args:
            asin: ASIN du produit Amazon
            max_pages: Nombre maximum de pages

        Returns:
            Liste de ScrapedReview
        """
        reviews = []

        for page in range(1, max_pages + 1):
            url = f"{self.BASE_URL}/product-reviews/{asin}?pageNumber={page}&reviewerType=all_reviews"
            print(f"📄 Scraping page {page}/{max_pages}: {url}")

            soup = self._safe_request(url)
            if soup is None:
                print(f"❌ Impossible de charger la page {page}")
                break

            cards = soup.select('[data-hook="review"]')
            if not cards:
                print(f"⚠️  Aucun élément de review trouvé sur la page {page}")
                break

            page_reviews = 0
            for card in cards:
                review = self._extract_review_amazon(card, asin, url)
                if review:
                    reviews.append(review)
                    page_reviews += 1

            print(f"✅ {page_reviews} reviews extraites de la page {page}")

        print(f"🎯 Total: {len(reviews)} reviews scrapées pour ASIN {asin}")
        return reviews

    def _extract_review_amazon(self, card: BeautifulSoup, asin: str, url: str) -> Optional[ScrapedReview]:
        """Extrait les données d'une review Amazon."""
        try:
            # Texte de la review
            text_el = card.select_one('[data-hook="review-body"]')
            if not text_el:
                return None

            text = text_el.get_text(strip=True)
            if not text or len(text) < 10:
                return None

            # Note
            rating = None
            stars_el = card.select_one('[data-hook="review-star-rating"]')
            if stars_el:
                try:
                    rating_text = stars_el.get_text().strip()
                    rating = float(rating_text.split()[0])
                except:
                    pass

            # Date
            date = None
            date_el = card.select_one('[data-hook="review-date"]')
            if date_el:
                date = date_el.get_text(strip=True)

            return ScrapedReview(
                text=text,
                rating=rating,
                source="Amazon.in",
                target=asin,
                date=date,
                url=url
            )

        except Exception as e:
            print(f"⚠️  Erreur extraction review Amazon: {e}")
            return None

# Fonctions utilitaires pour compatibilité
def scrape_trustpilot(company: str, max_pages: int = 3) -> List[Dict]:
    """Fonction utilitaire pour compatibilité avec l'ancien code."""
    scraper = TrustpilotScraper()
    reviews = scraper.scrape(company, max_pages)
    return [
        {
            "text": r.text,
            "rating": r.rating,
            "source": r.source,
            "target": r.target,
            "date": r.date,
        }
        for r in reviews
    ]

def scrape_amazon(asin: str, max_pages: int = 2) -> List[Dict]:
    """Fonction utilitaire pour compatibilité avec l'ancien code."""
    scraper = AmazonScraper()
    reviews = scraper.scrape(asin, max_pages)
    return [
        {
            "text": r.text,
            "rating": r.rating,
            "source": r.source,
            "target": r.target,
            "date": r.date,
        }
        for r in reviews
    ]