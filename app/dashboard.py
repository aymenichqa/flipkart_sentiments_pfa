# app/dashboard.py
# Lancer avec : streamlit run app/dashboard.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://localhost:8000"  # changer si déployé

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Flipkart Sentiment Analysis — Dashboard")
st.markdown("Analysez le sentiment de reviews e-commerce en temps réel.")

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📝 Texte libre", "🌐 Scraping", "📊 Batch"])

# ── Tab 1 : Prédiction texte libre ───────────────────────────────────────────
with tab1:
    st.subheader("Analyser un avis unique")
    user_text = st.text_area(
        "Entrez un avis client en anglais :",
        placeholder="e.g. Great product! Fast delivery and excellent quality.",
        height=120
    )
    if st.button("Analyser", type="primary"):
        if user_text.strip():
            with st.spinner("Analyse en cours..."):
                resp = requests.post(f"{API_URL}/predict", json={"text": user_text})
                result = resp.json()

            color = {"POSITIF": "🟢", "NEUTRE": "🟡", "NEGATIF": "🔴"}
            st.markdown(f"## {color[result['sentiment']]} {result['sentiment']}")
            st.metric("Confiance", f"{result['confidence']*100:.1f}%")

            # Jauge de probabilités
            fig = go.Figure(go.Bar(
                x=list(result['probabilities'].values()),
                y=list(result['probabilities'].keys()),
                orientation='h',
                marker_color=['#27ae60', '#f39c12', '#e74c3c']
            ))
            fig.update_layout(title="Probabilités par classe", xaxis_range=[0, 1],
                              height=200, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 : Scraping ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Scraper et analyser une entreprise")

    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("Source", ["trustpilot", "amazon"])
    with col2:
        target = st.text_input(
            "Entreprise / ASIN",
            placeholder="amazon.in" if source == "trustpilot" else "B08N5WRWNW"
        )
    with col3:
        max_pages = st.slider("Pages à scraper", 1, 5, 2)

    if st.button("🚀 Scraper & Analyser", type="primary"):
        if target.strip():
            with st.spinner(f"Scraping {source}/{target} ({max_pages} pages)..."):
                resp = requests.post(f"{API_URL}/scrape", json={
                    "source": source, "target": target, "max_pages": max_pages
                })

            if resp.status_code == 200:
                data  = resp.json()
                stats = data["statistics"]
                reviews = data["reviews"]

                # Métriques globales
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total reviews", stats["total"])
                col2.metric("✅ Positif", f"{stats['POSITIF']} ({stats['pct_positif']}%)")
                col3.metric("❌ Négatif", f"{stats['NEGATIF']} ({stats['pct_negatif']}%)")
                col4.metric("Confiance moy.", f"{stats['confidence_moyenne']*100:.1f}%")

                if stats["alerte"]:
                    st.error("⚠️ ALERTE : Plus de 30% de reviews négatives !")

                # Graphiques
                col_a, col_b = st.columns(2)

                with col_a:
                    df_counts = pd.DataFrame({
                        "Sentiment": ["POSITIF", "NEUTRE", "NEGATIF"],
                        "Count": [stats["POSITIF"], stats["NEUTRE"], stats["NEGATIF"]]
                    })
                    fig = px.pie(df_counts, values="Count", names="Sentiment",
                                 color="Sentiment",
                                 color_discrete_map={"POSITIF": "#27ae60",
                                                     "NEUTRE": "#f39c12",
                                                     "NEGATIF": "#e74c3c"},
                                 title="Distribution des sentiments")
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    df_rev = pd.DataFrame(reviews)
                    fig = px.histogram(df_rev, x="confidence", color="sentiment",
                                       color_discrete_map={"POSITIF": "#27ae60",
                                                           "NEUTRE": "#f39c12",
                                                           "NEGATIF": "#e74c3c"},
                                       nbins=20,
                                       title="Distribution de la confiance")
                    st.plotly_chart(fig, use_container_width=True)

                # Tableau des reviews
                st.subheader("Détail des reviews")
                df_display = pd.DataFrame(reviews)[
                    ["text", "sentiment", "confidence", "rating", "date"]
                ]
                df_display["confidence"] = df_display["confidence"].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                st.dataframe(df_display, use_container_width=True)

                # Export CSV
                csv = pd.DataFrame(reviews).to_csv(index=False)
                st.download_button("⬇️ Télécharger CSV", csv,
                                   f"results_{target}_{source}.csv", "text/csv")
            else:
                st.error(f"Erreur : {resp.json().get('detail', 'Inconnue')}")

# ── Tab 3 : Batch ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Analyser plusieurs avis en batch")
    uploaded = st.file_uploader("Charger un CSV avec une colonne 'review'", type="csv")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if "review" in df_up.columns:
            texts = df_up["review"].dropna().tolist()[:100]
            if st.button(f"Analyser {len(texts)} reviews"):
                with st.spinner("Analyse batch en cours..."):
                    resp = requests.post(f"{API_URL}/predict/batch", json={"texts": texts})
                    data = resp.json()

                st.success(f"✅ {data['statistics']['total']} reviews analysées")
                col1, col2, col3 = st.columns(3)
                col1.metric("POSITIF", data['statistics']['POSITIF'])
                col2.metric("NEUTRE",  data['statistics']['NEUTRE'])
                col3.metric("NEGATIF", data['statistics']['NEGATIF'])

                df_results = pd.DataFrame(data["results"])
                st.dataframe(df_results[["text_original", "sentiment", "confidence"]])
                