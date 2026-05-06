# app/dashboard.py — Dashboard Soutenance V3 (Mode "Waouh")
# ══════════════════════════════════════════════════════════════════════════════
# Lancer : streamlit run app/dashboard.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Configuration globale ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plateforme Sentiment Maroc",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── CSS personnalisé — rendu "Waouh" ─────────────────────────────────────────
st.markdown("""
<style>
    /* Fond général légèrement plus sombre */
    .stApp { background-color: #0f1117; }

    /* Carte de résultat sentiment */
    .result-card {
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin: 8px 0;
        border: 2px solid;
    }
    .result-positif { background: #0a2e1a; border-color: #2ecc71; color: #2ecc71; }
    .result-neutre  { background: #2e1f00; border-color: #f39c12; color: #f39c12; }
    .result-negatif { background: #2e0a0a; border-color: #e74c3c; color: #e74c3c; }

    /* Badge modèle */
    .badge-v1 {
        background: #1a3a5c; color: #5dade2;
        border-radius: 6px; padding: 4px 10px;
        font-size: 12px; font-weight: bold;
    }
    .badge-v2 {
        background: #3a1a5c; color: #a855f7;
        border-radius: 6px; padding: 4px 10px;
        font-size: 12px; font-weight: bold;
    }

    /* KPI cards */
    .kpi-card {
        background: #1a1d24;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2a2d35;
    }
    .kpi-value { font-size: 28px; font-weight: 700; }
    .kpi-label { font-size: 12px; color: #888; margin-top: 4px; }

    /* Alerte négative */
    .alerte-box {
        background: #3b0d0d;
        border: 1.5px solid #e74c3c;
        border-radius: 8px;
        padding: 12px 16px;
        color: #e74c3c;
        font-weight: bold;
        margin: 10px 0;
    }

    /* Separator stylisé */
    .separator { border-top: 1px solid #2a2d35; margin: 16px 0; }

    /* Sidebar radio custom */
    div[data-testid="stRadio"] > label { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ── Constantes visuelles ──────────────────────────────────────────────────────
COLORS  = {"POSITIF": "#2ecc71", "NEUTRE": "#f39c12", "NEGATIF": "#e74c3c"}
ICONS   = {"POSITIF": "😊",      "NEUTRE": "😐",       "NEGATIF": "😠"}
CSS_CLS = {"POSITIF": "result-positif", "NEUTRE": "result-neutre", "NEGATIF": "result-negatif"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def badge_modele(version: str) -> str:
    if version == "v2":
        return '<span class="badge-v2">🚀 XLM-RoBERTa V2</span>'
    return '<span class="badge-v1">📊 TF-IDF V1</span>'

def proba_gauge(probas: dict) -> go.Figure:
    """Graphique horizontal des probabilités — design épuré."""
    labels = list(probas.keys())
    values = [v * 100 for v in probas.values()]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=[COLORS.get(l, "#888") for l in labels],
            line=dict(width=0),
        ),
        text=[f"<b>{v:.1f}%</b>" for v in values],
        textposition="outside",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color="white")),
        showlegend=False,
    )
    return fig

def pie_chart(stats: dict) -> go.Figure:
    """Graphique circulaire — répartition des sentiments."""
    labels  = ["POSITIF", "NEUTRE", "NEGATIF"]
    values  = [stats.get("POSITIF", 0), stats.get("NEUTRE", 0), stats.get("NEGATIF", 0)]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(
            colors=["#2ecc71", "#f39c12", "#e74c3c"],
            line=dict(color="#0f1117", width=3),
        ),
        textinfo="percent+label",
        textfont=dict(size=13),
        hovertemplate="%{label}: %{value} avis (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="v",
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Centre de Contrôle
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🇲🇦 Plateforme Sentiment")
    st.markdown("Veille e-commerce & local — Maroc")
    st.divider()

    # ── Statut API ────────────────────────────────────────────────────────────
    try:
        h = requests.get(f"{API_BASE}/health", timeout=3).json()
        v1_ok = h.get("modeles", {}).get("v1", {}).get("disponible", False)
        v2_ok = h.get("modeles", {}).get("v2", {}).get("disponible", False)
        st.success("✅ API connectée")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("V1 TF-IDF", "✅" if v1_ok else "❌")
        col_s2.metric("V2 RoBERTa", "✅" if v2_ok else "⏳")
    except Exception:
        v1_ok, v2_ok = False, False
        st.error("❌ API hors ligne\n\n`uvicorn app.main:app --reload`")

    st.divider()

    # ── Sélecteur de modèle (le cœur de l'A/B testing) ───────────────────────
    st.markdown("### 🤖 Choisir le Modèle IA")
    model_choice = st.radio(
        label="Modèle actif :",
        options=["v1", "v2"],
        format_func=lambda v: (
            "📊 V1 — TF-IDF (Rapide / Classique)"
            if v == "v1"
            else "🚀 V2 — XLM-RoBERTa (Deep Learning Multilingue)"
        ),
        index=1,    # V2 sélectionné par défaut
        key="model_choice_radio",
    )

    st.divider()

    # ── Description du modèle sélectionné ────────────────────────────────────
    if model_choice == "v1":
        st.info(
            "**📊 Modèle V1 — TF-IDF**\n\n"
            "- Approche classique (sac de mots)\n"
            "- Rapide (~1ms / prédiction)\n"
            "- ⚠️ Anglais uniquement\n"
            "- ⚠️ Ne comprend pas la négation\n"
            "- ⚠️ Sarcasme = piège"
        )
    else:
        st.success(
            "**🚀 Modèle V2 — XLM-RoBERTa**\n\n"
            "- Deep Learning multilingue\n"
            "- FR · AR · Darija · EN...\n"
            "- ✅ Comprend le contexte\n"
            "- ✅ Gère la négation\n"
            "- ✅ Sarcasme détecté"
        )

    st.divider()
    st.caption("PFA 2025–2026 · Analyse de Sentiment")


# ══════════════════════════════════════════════════════════════════════════════
# TITRE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<h1 style='text-align:center;margin-bottom:4px'>🇲🇦 Plateforme Analyse de Sentiment</h1>"
    f"<p style='text-align:center;color:#888;margin-bottom:24px'>"
    f"Modèle actif : {badge_modele(model_choice)}"
    f"</p>",
    unsafe_allow_html=True,
)

# ── Tabs principaux ───────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🧪 Crash Test IA",
    "🔍 Veille Réelle (Scraping)",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Crash Test IA (A/B Testing manuel)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Testez les deux modèles en direct")
    st.markdown(
        "Entrez un avis en **Français**, **Darija** ou **Anglais**. "
        "Basculez entre V1 et V2 dans la sidebar pour voir la différence."
    )

    # ── Exemples pour démonstration soutenance ────────────────────────────────
    col_ex_title, col_ex_select = st.columns([1, 2])
    with col_ex_title:
        st.markdown("**Phrases de test :**")
    with col_ex_select:
        exemples = {
            "🇫🇷 Positif clair"         : "Ce produit est vraiment excellent ! Livraison rapide et qualité irréprochable.",
            "🇫🇷 Négatif clair"         : "Très déçu, produit de mauvaise qualité. Arnaque totale !",
            "🇫🇷 Sarcastique (piège V1)" : "Oh super, le produit est arrivé en mille morceaux. Vraiment magnifique emballage...",
            "🇫🇷 Négatif subtil (piège V1)": "Ce n'est pas vraiment terrible comme produit, je ne suis pas du tout satisfait.",
            "Darija — Positif"           : "Mezian bzzaf had produit, livraison sari3a w machi ghalya. Kan nsawb merci !",
            "Darija — Négatif"           : "Machi mezian, 3tani chwi khaybat. Ma kench kolchi b7al f soura.",
            "Darija + FR (mixte)"        : "Service correct mais la qualité ma3endach, kayn mochkil m3a packaging.",
            "🇬🇧 Negative EN"            : "Absolutely terrible. Broke after 2 days. Complete waste of money.",
        }
        exemple_key = st.selectbox("Charger un exemple :", list(exemples.keys()), label_visibility="collapsed")

    user_text = st.text_area(
        "Votre avis :",
        value=exemples[exemple_key],
        height=110,
        placeholder="Entrez un avis en français, darija ou anglais...",
        key="text_input_tab1",
    )

    col_btn, col_model_info = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)
    with col_model_info:
        st.markdown(
            f"Modèle : {badge_modele(model_choice)}",
            unsafe_allow_html=True
        )

    if analyze_btn:
        if not user_text.strip():
            st.warning("⚠️ Veuillez entrer un texte.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        json={"text": user_text, "model_version": model_choice},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        r   = resp.json()
                        s   = r["sentiment"]
                        c   = r["confidence"]
                        mod = r.get("modele", "")
                        pd  = r.get("probabilities", {})

                        st.divider()
                        col_r1, col_r2, col_r3 = st.columns([1, 1, 2])

                        # ── Résultat principal ────────────────────────────────
                        with col_r1:
                            st.markdown(
                                f"<div class='result-card {CSS_CLS[s]}'>"
                                f"<div style='font-size:2.5rem'>{ICONS[s]}</div>"
                                f"<div style='font-size:1.6rem;font-weight:800'>{s}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        # ── Métriques ─────────────────────────────────────────
                        with col_r2:
                            st.metric("Confiance", f"{c*100:.1f}%")
                            st.markdown(
                                f"Modèle : {badge_modele(model_choice)}",
                                unsafe_allow_html=True,
                            )
                            # Nom exact du modèle
                            st.caption(f"🔬 {mod}")

                        # ── Graphique probabilités ────────────────────────────
                        with col_r3:
                            if pd:
                                st.plotly_chart(proba_gauge(pd), use_container_width=True)

                        # ── Conseil pédagogique pour la soutenance ────────────
                        if model_choice == "v1" and "sarcast" in exemple_key.lower():
                            st.markdown(
                                "<div class='alerte-box'>"
                                "💡 <b>Point clé soutenance :</b> Le modèle V1 TF-IDF voit des mots positifs "
                                "(\"magnifique\", \"super\") et prédit POSITIF malgré l'intention sarcastique. "
                                "Passez en V2 (XLM-RoBERTa) pour voir la différence — le Transformer comprend le contexte global."
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        elif model_choice == "v1" and "Darija" in exemple_key:
                            st.markdown(
                                "<div class='alerte-box'>"
                                "💡 <b>Limite V1 :</b> TF-IDF est entraîné sur de l'anglais. "
                                "La Darija n'est pas dans son vocabulaire. Passez en V2 pour un résultat pertinent."
                                "</div>",
                                unsafe_allow_html=True,
                            )

                    else:
                        err = resp.json().get("detail", "Erreur inconnue")
                        if "V2" in err or "XLM" in err:
                            st.error(f"❌ {err}")
                            st.info("Installer V2 : `pip install transformers torch sentencepiece`")
                        else:
                            st.error(f"Erreur API : {err}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ API non joignable. Lancer : `uvicorn app.main:app --reload`")
                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout. V2 peut prendre 30–60s la première fois (téléchargement du modèle).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Veille Réelle (Scraping Marocain)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Veille E-commerce & Local — Sources Marocaines")
    st.markdown(
        f"Scraping + Analyse IA avec : {badge_modele(model_choice)}",
        unsafe_allow_html=True
    )

    # ── Formulaire de scraping ────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([1, 2])

    with col_f1:
        source = st.selectbox(
            "📡 Source",
            options=["jumia", "marjane", "gmaps"],
            format_func=lambda s: {
                "jumia"  : "🛍️  Jumia Maroc (jumia.ma)",
                "marjane": "🏪 Marjane Mall (marjanemall.ma)",
                "gmaps"  : "📍 Google Maps (Commerces locaux)",
            }[s],
        )

    placeholders = {
        "jumia"  : "https://www.jumia.ma/generic-samsung-galaxy-a15/mpXXXXXXX/reviews/",
        "marjane": "https://marjanemall.ma/produits/telephone-portable-xyz",
        "gmaps"  : "https://www.google.com/maps/place/Marjane+Hay+Riad/@33.9...",
    }
    tips = {
        "jumia"  : "💡 Sur Jumia, aller sur la page produit → cliquer 'Avis' → copier l'URL (se termine par /reviews/)",
        "marjane": "💡 Copier l'URL directe de la page produit sur marjanemall.ma",
        "gmaps"  : "💡 Sur Google Maps : chercher le commerce → Partager → Copier le lien",
    }

    with col_f2:
        target_url = st.text_input(
            "🔗 URL cible",
            placeholder=placeholders[source],
            help=tips[source],
        )
        st.caption(tips[source])

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if source == "gmaps":
            target_reviews = st.slider("🎯 Nombre d'avis cibles (scroll)", 5, 50, 20, step=5)
            max_pages = 1
        else:
            max_pages = st.slider("📄 Pages à scraper", 1, 5, 2)
            target_reviews = 20
    with col_p2:
        st.markdown("")  # espaceur
        scrape_btn = st.button(
            "🚀 Lancer le Scraping & Analyser",
            type="primary",
            use_container_width=True,
        )

    # ── Résultats du scraping ─────────────────────────────────────────────────
    if scrape_btn:
        if not target_url.strip():
            st.warning("⚠️ Veuillez renseigner une URL.")
        else:
            progress_msg = {
                "jumia"  : f"Scraping Jumia ({max_pages} pages) + analyse {model_choice.upper()}...",
                "marjane": f"Scraping Marjane Mall (networkidle + {max_pages} pages)...",
                "gmaps"  : f"Scraping Google Maps ({target_reviews} avis, infinite scroll)...",
            }
            with st.spinner(progress_msg[source]):
                try:
                    payload = {
                        "source"        : source,
                        "url"           : target_url,
                        "model_version" : model_choice,
                        "max_pages"     : max_pages,
                        "target_reviews": target_reviews,
                    }
                    resp = requests.post(
                        f"{API_BASE}/scrape",
                        json=payload,
                        timeout=240,   # scraping peut être long (Playwright)
                    )

                    if resp.status_code == 200:
                        data    = resp.json()
                        stats   = data["statistics"]
                        reviews = data["reviews"]
                        df      = pd.DataFrame(reviews)

                        st.divider()

                        # ── Ligne de titre ────────────────────────────────────
                        st.markdown(
                            f"#### Résultats — {source.upper()} · "
                            f"{badge_modele(model_choice)}",
                            unsafe_allow_html=True,
                        )

                        # ── Alerte négative ───────────────────────────────────
                        if stats.get("alerte_negative"):
                            st.markdown(
                                "<div class='alerte-box'>"
                                "🚨 ALERTE — Plus de 30% d'avis négatifs détectés ! "
                                "Action recommandée : vérifier la qualité du produit/service."
                                "</div>",
                                unsafe_allow_html=True,
                            )

                        # ── KPIs (st.metric) ──────────────────────────────────
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("📋 Total avis",  stats.get("total", 0))
                        k2.metric(
                            "😊 Positifs",
                            f"{stats.get('POSITIF', 0)}",
                            delta=f"{stats.get('pct_positif', 0)}%",
                        )
                        k3.metric(
                            "😐 Neutres",
                            f"{stats.get('NEUTRE', 0)}",
                            delta=f"{stats.get('pct_neutre', 0)}%",
                            delta_color="off",
                        )
                        k4.metric(
                            "😠 Négatifs",
                            f"{stats.get('NEGATIF', 0)}",
                            delta=f"-{stats.get('pct_negatif', 0)}%",
                            delta_color="inverse",
                        )
                        k5.metric(
                            "🎯 Confiance moy.",
                            f"{stats.get('confidence_moyenne', 0)*100:.1f}%",
                        )

                        # ── Graphiques ────────────────────────────────────────
                        col_g1, col_g2 = st.columns([1, 1])

                        with col_g1:
                            st.plotly_chart(
                                pie_chart(stats),
                                use_container_width=True,
                            )

                        with col_g2:
                            # Graphique confiance par sentiment
                            if "confidence" in df.columns and "sentiment" in df.columns:
                                conf_df = df.groupby("sentiment")["confidence"].agg(
                                    ["mean", "min", "max"]
                                ).reset_index()
                                fig_conf = go.Figure()
                                for _, row in conf_df.iterrows():
                                    sent = row["sentiment"]
                                    fig_conf.add_trace(go.Bar(
                                        name=sent,
                                        x=[sent],
                                        y=[row["mean"]],
                                        marker_color=COLORS.get(sent, "#888"),
                                        text=f"{row['mean']*100:.1f}%",
                                        textposition="outside",
                                    ))
                                fig_conf.update_layout(
                                    title="Confiance moyenne / Sentiment",
                                    yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                                    height=280,
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    showlegend=False,
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    font=dict(color="white"),
                                )
                                st.plotly_chart(fig_conf, use_container_width=True)

                        # ── Tableau des avis avec st.expander ─────────────────
                        st.markdown("#### 📋 Détail des avis analysés")

                        for sentiment in ["NEGATIF", "POSITIF", "NEUTRE"]:
                            df_sent = df[df["sentiment"] == sentiment]
                            if df_sent.empty:
                                continue
                            icon  = ICONS[sentiment]
                            count = len(df_sent)
                            color = COLORS[sentiment]

                            with st.expander(
                                f"{icon} {sentiment} — {count} avis",
                                expanded=(sentiment == "NEGATIF"),
                            ):
                                for _, row in df_sent.iterrows():
                                    col_avis, col_meta = st.columns([4, 1])
                                    with col_avis:
                                        st.markdown(
                                            f"<div style='padding:8px 12px;"
                                            f"border-left:3px solid {color};"
                                            f"margin-bottom:8px;border-radius:0 6px 6px 0'>"
                                            f"{row.get('text', '')}"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                    with col_meta:
                                        rating = row.get("rating")
                                        if rating:
                                            stars = "⭐" * int(rating)
                                            st.caption(stars)
                                        conf = row.get("confidence", 0)
                                        st.caption(f"Confiance : {conf*100:.0f}%")
                                        if row.get("date"):
                                            st.caption(f"📅 {row['date']}")

                        # ── Export CSV ────────────────────────────────────────
                        cols_export = [c for c in
                                       ["text", "sentiment", "confidence", "rating", "date", "source"]
                                       if c in df.columns]
                        csv = df[cols_export].to_csv(index=False, encoding="utf-8-sig")
                        st.download_button(
                            "⬇️ Exporter les résultats (CSV)",
                            data=csv,
                            file_name=f"sentiments_{source}_{model_choice}.csv",
                            mime="text/csv",
                        )

                    else:
                        err = resp.json().get("detail", "Erreur inconnue")
                        st.error(f"❌ Erreur API : {err}")
                        if "503" in str(resp.status_code):
                            st.info("V2 non chargé : `pip install transformers torch sentencepiece`")

                except requests.exceptions.ConnectionError:
                    st.error("❌ API non joignable. Lancer : `uvicorn app.main:app --reload`")
                except requests.exceptions.Timeout:
                    st.error(
                        "⏰ Timeout (240s dépassé).\n\n"
                        "Le scraping Playwright peut être lent. "
                        "Réduire le nombre de pages ou d'avis cibles."
                    )
                except Exception as e:
                    st.error(f"Erreur inattendue : {e}")