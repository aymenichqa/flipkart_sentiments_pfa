# app/dashboard.py — Dashboard Soutenance V6 (Simple & Efficace)
# ═══════════════════════════════════════════════════════════════════════
# Lancer : streamlit run app/dashboard.py
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import os
import json
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentLab — Analyse de Sentiments",
    page_icon="🧠",
    layout="wide",
)

API_BASE = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")

# ═══════════════════════════════════════════════════════════════════════
# CSS — Design Propre & Clair (pas de dock, pas de bento excessif)
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: #0f172a;
        color: #e2e8f0;
    }
    
    /* Cards propres */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }
    
    .card-title {
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    
    /* Bouton principal */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Inputs */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Timeline */
    .timeline-item {
        border-left: 2px solid rgba(255, 255, 255, 0.08);
        padding-left: 20px;
        padding-bottom: 20px;
        position: relative;
    }
    
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -5px;
        top: 4px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6366f1;
    }
    
    .timeline-item.positif::before { background: #10b981; }
    .timeline-item.negatif::before { background: #f43f5e; }
    
    /* Hide streamlit stuff */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* DataFrame */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# BASE DE DONNÉES (RECOMMENCÉE À ZÉRO)
# ═══════════════════════════════════════════════════════════════════════

def init_db():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # NOUVELLE TABLE avec v2_model_key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            texte_ou_url TEXT NOT NULL,
            modele_utilise TEXT NOT NULL,
            v2_model_key TEXT,
            sentiment TEXT,
            confiance REAL,
            source TEXT,
            type_analyse TEXT NOT NULL
        )
    """)
    # Migration : corrige les anciennes entrées où sentiment était NULL (NaN)
    cursor.execute("""
        UPDATE logs SET sentiment = '' WHERE sentiment IS NULL OR sentiment = 'None'
    """)
    conn.commit()
    conn.close()

def save_to_db(texte_ou_url, modele_utilise, v2_model_key=None, sentiment=None, confiance=None, source=None, type_analyse="prediction"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO logs (date, texte_ou_url, modele_utilise, v2_model_key, sentiment, confiance, source, type_analyse)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), texte_ou_url, modele_utilise, v2_model_key, sentiment, confiance, source, type_analyse))
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY date DESC", conn)
    conn.close()
    return df

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM logs")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT modele_utilise, v2_model_key, COUNT(*) as cnt FROM logs GROUP BY modele_utilise, v2_model_key ORDER BY cnt DESC LIMIT 1")
    most_used = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM logs WHERE type_analyse='prediction'")
    total_predictions = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM logs WHERE type_analyse='scraping'")
    total_scrapings = cursor.fetchone()[0]
    conn.close()
    return {
        "total": total,
        "most_used_model": f"{most_used[0]}-{most_used[1]}" if most_used and most_used[1] else (most_used[0] if most_used else "—"),
        "total_predictions": total_predictions,
        "total_scrapings": total_scrapings
    }

init_db()

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

COLORS = {"POSITIF": "#10B981", "NEUTRE": "#818CF8", "NEGATIF": "#F43F5E"}
ICONS = {"POSITIF": "✨", "NEUTRE": "◉", "NEGATIF": "◉"}

def show_model_badge(version, v2_key=None):
    """Affiche un badge de modèle (version V1 ou V2 avec sa variante)."""
    if version == "v1":
        st.markdown("📊 **TF-IDF V1** — Anglais uniquement")
        return
    names = {"default": "🎯 XLM-RoBERTa Default", "twitter": "🐦 XLM-RoBERTa Twitter", "french": "🥖 XLM-RoBERTa French"}
    st.markdown(f"**{names.get(v2_key or 'default', '🚀 V2')}**")

def show_sentiment_result(sentiment, confidence, probas=None):
    """Affiche le résultat du sentiment avec métriques et barres de progression natives."""
    icon = ICONS.get(sentiment, "◉")
    color = COLORS.get(sentiment, "#888")
    
    # Carte résultat avec métrique principale
    st.markdown(f"""
    <div style="text-align: center; padding: 30px 20px; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 16px;">
        <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 28px; font-weight: 700; color: {color};">{sentiment}</div>
        <div style="font-size: 14px; color: #64748b; margin-top: 4px;">Confiance : {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Barres de progression natives Streamlit
    if probas:
        st.markdown("**📊 Distribution des probabilités**")
        for label, val in probas.items():
            cols = st.columns([1, 4, 1])
            with cols[0]:
                st.markdown(f"**{label}**")
            with cols[1]:
                st.progress(val, text="")
            with cols[2]:
                st.markdown(f"**{val*100:.1f}%**")

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align: center; margin-bottom: 32px;">
    <h1 style="font-size: 2.5rem; font-weight: 700; color: #fff; margin-bottom: 4px;">SentimentLab</h1>
    <p style="font-size: 13px; color: #64748b; letter-spacing: 2px; text-transform: uppercase;">Analyse de Sentiments Multilingue</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TABS CLASSIQUES (simples et efficaces)
# ═══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🧪 Test en Direct", "🕸️ Veille Scraping", "📁 Archives"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 : TEST EN DIRECT
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="card-title">Configuration du Modèle</div>', unsafe_allow_html=True)
    
    # ── Choix du modèle V2 (CLÉ !) ─────────────────────────────────────
    col_model, col_info = st.columns([1, 2])
    
    with col_model:
        model_version = st.radio(
            "Version du modèle",
            ["V2 — XLM-RoBERTa", "V1 — TF-IDF (Anglais uniquement)"],
            index=0,
            help="V2 = multilingue (FR/AR/Darija). V1 = rapide mais anglais uniquement."
        )
        
        if "V2" in model_version:
            v2_model_key = st.selectbox(
                "Variante V2",
                ["twitter", "default", "french"],
                format_func=lambda k: {
                    "twitter": "🐦 Twitter — Textes courts / Slang / Emojis",
                    "default": "🎯 Default — Avis longs / E-commerce",
                    "french": "🥖 French — Français formel"
                }[k],
                help="Choisissez selon le type de texte à analyser"
            )
        else:
            v2_model_key = None
    
    with col_info:
        if v2_model_key == "twitter":
            st.info("**🐦 Twitter** — Optimisé pour : commentaires courts, slang, emojis, sarcasme. **Idéal pour Jumia & réseaux sociaux.**")
        elif v2_model_key == "default":
            st.info("**🎯 Default** — Optimisé pour : avis longs, e-commerce, reviews détaillées. **Idéal pour Trustpilot & TripAdvisor.**")
        elif v2_model_key == "french":
            st.info("**🥖 French** — Optimisé pour : français standard, syntaxe complexe. **Idéal pour Google Maps & emails.**")
        else:
            st.info("**📊 V1** — Rapide mais limité à l'anglais. Ne comprend pas la Darija.")
    
    st.markdown("---")
    
    # ── Zone de texte ──────────────────────────────────────────────────
    col_input, col_result = st.columns([1.2, 1], gap="large")
    
    with col_input:
        st.markdown('<div class="card-title">Texte à Analyser</div>', unsafe_allow_html=True)
        
        exemples = {
            "✏️ Texte libre": "",
            "🇫🇷 Positif (FR)": "Ce produit est vraiment excellent ! Livraison rapide, je recommande vivement.",
            "🇫🇷 Négatif (FR)": "Très déçu, qualité médiocre et service client inexistant.",
            "🇲🇦 Darija positive": "Mzyan bzzaf had l'produit, srite men Jumia w wasalni f 2 jours !",
            "🇲🇦 Darija négative": "Khayb had l'produit, mafhemch chno dert b had l flous, 7ram.",
            "😂 Sarcasme": "Oh super, encore un chargeur qui marche 2 jours. Quelle qualité !",
            "😍 Emoji social": "😍😍 j'adore vraiment trop bien ce téléphone omg",
        }
        
        exemple = st.selectbox("Charger un exemple", list(exemples.keys()), label_visibility="collapsed")
        user_text = st.text_area("Votre texte", value=exemples[exemple], height=140, 
                                placeholder="Collez un avis client ici...", label_visibility="collapsed")
        
        # Option comparaison
        compare_all = st.toggle("🔬 Comparer les 3 modèles V2", value=False,
                               help="Lance l'analyse sur les 3 variantes simultanément")
        
        analyze_btn = st.button("🔮 Analyser le Sentiment", type="primary", use_container_width=True)
    
    # ── Résultat ───────────────────────────────────────────────────────
    with col_result:
        if analyze_btn and user_text.strip():
            with st.spinner("Analyse en cours..."):
                try:
                    if compare_all and "V2" in model_version:
                        # Comparaison des 3 modèles
                        results = {}
                        for key in ["default", "twitter", "french"]:
                            resp = requests.post(f"{API_BASE}/predict", 
                                               json={"text": user_text, "model_version": "v2", "v2_model_key": key},
                                               timeout=60)
                            if resp.status_code == 200:
                                results[key] = resp.json()
                        
                        # Sauvegarde du meilleur
                        best = max(results.values(), key=lambda x: x["confidence"])
                        save_to_db(user_text[:200], "v2", best.get("v2_model_key"), best["sentiment"], best["confidence"], None, "prediction")
                        
                        # Affichage comparaison
                        st.markdown("**📊 Comparaison des Modèles**")
                        
                        cols = st.columns(3)
                        model_info = {
                            "default": ("Default", "🎯"),
                            "twitter": ("Twitter", "🐦"),
                            "french": ("French", "🥖")
                        }
                        
                        for i, (key, res) in enumerate(results.items()):
                            name, icon = model_info[key]
                            is_best = res == best
                            with cols[i]:
                                if is_best:
                                    st.success(f"🏆 **BEST**")
                                st.markdown(f"{icon} **{name}**")
                                st.metric("Sentiment", res['sentiment'])
                                st.metric("Confiance", f"{res['confidence']*100:.0f}%")
                                with st.expander("Probabilités"):
                                    for label, val in res.get("probabilities", {}).items():
                                        st.markdown(f"**{label}** : {val*100:.1f}%")
                                        st.progress(val, text="")
                    
                    else:
                        # Analyse simple
                        mv = "v2" if "V2" in model_version else "v1"
                        payload = {"text": user_text, "model_version": mv}
                        if mv == "v2":
                            payload["v2_model_key"] = v2_model_key
                        
                        resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
                        
                        if resp.status_code == 200:
                            r = resp.json()
                            s, c = r["sentiment"], r["confidence"]
                            
                            save_to_db(user_text[:200], mv, r.get("v2_model_key"), s, c, None, "prediction")
                            
                            # Badge modèle
                            show_model_badge(mv, r.get("v2_model_key"))
                            
                            # Résultat
                            show_sentiment_result(s, c, r.get("probabilities", {}))
                            
                            # Info Darija
                            if r.get("darija_translated"):
                                st.success("🔄 Darija détectée et traduite automatiquement")
                
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            # État vide
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; opacity: 0.5;">
                <div style="font-size: 48px; margin-bottom: 16px;">🔮</div>
                <div style="font-size: 16px; font-weight: 600; color: #475569;">En attente d'analyse</div>
                <div style="font-size: 13px; color: #334155; margin-top: 8px;">Entrez un texte et cliquez sur Analyser</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 : VEILLE SCRAPING
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="card-title">Configuration du Scraping</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 2, 1.2, 1])
    
    with col1:
        SOURCE_ICONS = {"jumia": "🛒", "gmaps": "📍", "trustpilot": "⭐", "tripadvisor": "🗺️"}
        source = st.selectbox("Source", list(SOURCE_ICONS.keys()), 
                             format_func=lambda s: f"{SOURCE_ICONS[s]} {s.capitalize()}")
    
    with col2:
        placeholders = {
            "jumia": "https://www.jumia.ma/.../reviews/",
            "gmaps": "https://maps.app.goo.gl/...",
            "trustpilot": "https://www.trustpilot.com/review/...",
            "tripadvisor": "https://www.tripadvisor.com/Restaurant_Review-..."
        }
        target_url = st.text_input("URL", placeholder=placeholders[source])
    
    with col3:
        # MODÈLE V2 SELON LA SOURCE (recommandation intelligente)
        source_recommendations = {
            "jumia": "twitter",
            "gmaps": "french",
            "trustpilot": "default",
            "tripadvisor": "default"
        }
        default_model = source_recommendations.get(source, "twitter")
        
        scrape_model = st.selectbox(
            "Modèle V2",
            ["twitter", "default", "french"],
            index=["twitter", "default", "french"].index(default_model),
            format_func=lambda k: {
                "twitter": "🐦 Twitter (Jumia/RS)",
                "default": "🎯 Default (Trustpilot)",
                "french": "🥖 French (Google Maps)"
            }[k]
        )
        
        # Info contextuelle
        if source == "jumia":
            st.caption("💡 Twitter recommandé : commentaires courts + Darija")
        elif source == "gmaps":
            st.caption("💡 French recommandé : avis longs en FR")
        elif source in ["trustpilot", "tripadvisor"]:
            st.caption("💡 Default recommandé : avis détaillés multilingues")
    
    with col4:
        if source == "gmaps":
            target_reviews = st.number_input("Avis", 5, 50, 20, 5)
            max_pages = 1
        else:
            max_pages = st.number_input("Pages", 1, 5, 2)
            target_reviews = 20
    
    scrape_btn = st.button("🚀 Lancer l'Extraction", type="primary", use_container_width=True)
    
    if scrape_btn and target_url.strip():
        with st.spinner("Extraction en cours..."):
            try:
                resp = requests.post(f"{API_BASE}/scrape",
                                   json={"source": source, "url": target_url, "model_version": "v2",
                                         "v2_model_key": scrape_model, "max_pages": max_pages, "target_reviews": target_reviews},
                                   timeout=240)
                
                if resp.status_code == 200:
                    data = resp.json()
                    stats = data["statistics"]
                    df = pd.DataFrame(data["reviews"])
                    
                    save_to_db(target_url[:200], "v2", scrape_model, "SCRAPING", None, source, "scraping")
                    
                    # KPIs avec métriques natives
                    col_kpi = st.columns(4)
                    with col_kpi[0]: st.metric("Total Avis", stats.get("total", 0))
                    with col_kpi[1]: st.metric("Positifs", stats.get("POSITIF", 0))
                    with col_kpi[2]: st.metric("Neutres", stats.get("NEUTRE", 0))
                    with col_kpi[3]: st.metric("Négatifs", stats.get("NEGATIF", 0))
                    
                    # Alerte
                    if stats.get("alerte_negative"):
                        st.warning("🚨 Alerte : Plus de 30% d'avis négatifs détectés")
                    
                    # Graphiques
                    col_g1, col_g2 = st.columns([1, 1.5], gap="large")
                    
                    with col_g1:
                        fig = go.Figure(go.Pie(
                            labels=["POSITIF", "NEUTRE", "NEGATIF"],
                            values=[stats.get("POSITIF", 0), stats.get("NEUTRE", 0), stats.get("NEGATIF", 0)],
                            hole=0.6,
                            marker=dict(colors=["#10B981", "#818CF8", "#F43F5E"], line=dict(color="#0f172a", width=3)),
                            textinfo="percent", textfont=dict(size=13, color="white")
                        ))
                        fig.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_g2:
                        confidences = [r["confidence"] for r in data["reviews"]]
                        sentiments = [r["sentiment"] for r in data["reviews"]]
                        fig2 = px.histogram(x=confidences, color=sentiments, color_discrete_map=COLORS,
                                          nbins=20, labels={"x": "Confiance", "count": "Nombre"})
                        fig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         font=dict(color="#94a3b8"), showlegend=True,
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
                        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
                        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Tableau
                    st.markdown('<div class="card-title">Avis Analysés</div>', unsafe_allow_html=True)
                    display_df = df[["text", "sentiment", "confidence", "rating", "date"]].copy()
                    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(display_df, use_container_width=True, height=350)
            
            except Exception as e:
                st.error(f"Erreur : {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 : ARCHIVES
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="card-title">Historique des Analyses</div>', unsafe_allow_html=True)
    
    stats = get_stats()
    
    # Stats globales avec métriques natives
    col_stat = st.columns(4)
    with col_stat[0]: st.metric("Total", stats["total"])
    with col_stat[1]: st.metric("Prédictions", stats["total_predictions"])
    with col_stat[2]: st.metric("Scrapings", stats["total_scrapings"])
    with col_stat[3]: st.metric("Favori", stats["most_used_model"])
    
    # Export
    df_logs = get_all_logs()
    if not df_logs.empty:
        csv = df_logs.to_csv(index=False, encoding='utf-8')
        st.download_button("📥 Exporter CSV", csv, "sentimentlab_export.csv", "text/csv")
    
    # Timeline
    st.markdown('<div class="card-title">Dernières Analyses</div>', unsafe_allow_html=True)
    
    if not df_logs.empty:
        for _, row in df_logs.head(20).iterrows():
            sentiment = row.get("sentiment", "")
            sentiment = str(sentiment) if not isinstance(sentiment, str) else sentiment
            sentiment_class = sentiment.lower() if sentiment else "neutre"
            type_icon = "🧪" if row["type_analyse"] == "prediction" else "🕸️"
            date_str = datetime.fromisoformat(row["date"]).strftime("%d/%m/%Y %H:%M")
            
            st.markdown(f"""
            <div class="timeline-item {sentiment_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span style="font-size: 14px;">{type_icon}</span>
                        <span style="font-size: 11px; color: #64748b; text-transform: uppercase;">{row['type_analyse']}</span>
                        {f'<span style="font-size: 11px; font-weight: 600; color: {COLORS.get(sentiment, "#64748b")}; background: {COLORS.get(sentiment, "#64748b")}15; padding: 2px 10px; border-radius: 100px; border: 1px solid {COLORS.get(sentiment, "#64748b")}30;">{sentiment}</span>' if sentiment else ''}
                    </div>
                    <span style="font-size: 12px; color: #475569;">{date_str}</span>
                </div>
                <div style="font-size: 14px; color: #e2e8f0; margin-bottom: 4px;">{row['texte_ou_url'][:100]}{"..." if len(row['texte_ou_url']) > 100 else ""}</div>
                <div style="font-size: 12px; color: #475569;">
                    <span style="color: #64748b;">Modèle:</span> <span style="color: #94a3b8; font-weight: 500;">{row['modele_utilise'].upper()}</span>
                    {f' <span style="color: #64748b;">·</span> <span style="color: #64748b;">Confiance:</span> <span style="color: #94a3b8; font-weight: 500;">{row["confiance"]*100:.1f}%</span>' if row['confiance'] else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucune archive. Lancez une analyse pour commencer.")