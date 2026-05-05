from __future__ import annotations
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from config import MODEL_METRICS_FILE, PLOTS_DIR, MODELS_DIR

def build_app() -> None:
    st.set_page_config(page_title="Sephora - Prédiction de satisfaction client", layout="wide")

    st.title("💄 Sephora — Prédiction de la satisfaction client")
    st.markdown("Prédire si un produit Sephora sera bien noté (≥ 4/5) à partir de ses caractéristiques.")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Objectif", "📊 Données", "🤖 Modèles", "🔮 Prédiction"])

    with tab1:
        st.header("Objectif métier")
        st.write("""
        Sephora propose plus de 8 000 produits. L'objectif de ce projet est d'identifier 
        automatiquement les produits susceptibles d'obtenir une bonne note client (≥ 4/5), 
        à partir de leurs caractéristiques : prix, catégorie, type de peau ciblé, popularité.
        
        **Variable cible :** `good_rating` — 1 si note ≥ 4, 0 sinon.
        """)
        st.metric("Produits analysés", "8 494")
        st.metric("Avis clients", "1 094 411")
        st.metric("Accuracy meilleur modèle", "95.52%")

    with tab2:
        st.header("Insights sur les données")
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(PLOTS_DIR / "distribution_notes.png"), caption="Distribution des notes")
            st.image(str(PLOTS_DIR / "distribution_prix.png"), caption="Distribution des prix")
        with col2:
            st.image(str(PLOTS_DIR / "categories.png"), caption="Produits par catégorie")
            st.image(str(PLOTS_DIR / "matrices_confusion.png"), caption="Matrices de confusion")

    with tab3:
        st.header("Comparaison des modèles")
        st.image(str(PLOTS_DIR / "courbes_roc.png"), caption="Courbes ROC")
        if MODEL_METRICS_FILE.exists():
            metrics_df = pd.read_csv(MODEL_METRICS_FILE)
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Lance `python scripts/main.py` pour générer les métriques.")

    with tab4:
        st.header("Prédiction interactive")
        st.write("Entre les caractéristiques d'un produit pour prédire sa satisfaction client.")

        col1, col2 = st.columns(2)
        with col1:
            is_recommended = st.selectbox("Produit recommandé ?", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
            price = st.slider("Prix (USD)", 3, 500, 35)
            loves_count = st.number_input("Nombre de likes", 0, 500000, 1000)
            reviews = st.number_input("Nombre d'avis", 0, 10000, 50)
        with col2:
            skin_type = st.selectbox("Type de peau", [0, 1, 2, 3], format_func=lambda x: ["combination", "dry", "normal", "oily"][x])
            primary_category = st.selectbox("Catégorie", [0,1,2,3,4,5,6,7,8], format_func=lambda x: ["Bath & Body","Fragrance","Gifts","Hair","Makeup","Men","Mini Size","Skincare","Tools & Brushes"][x])
            new = st.selectbox("Nouveau produit ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            online_only = st.selectbox("Online only ?", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

        if st.button("Prédire"):
            try:
                gb = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
                features = np.array([[is_recommended, 0, 0, 0, price, loves_count, reviews, new, online_only, skin_type, primary_category, 0]])
                prediction = gb.predict(features)[0]
                proba = gb.predict_proba(features)[0][1]
                if prediction == 1:
                    st.success(f"✅ Bonne satisfaction prévue — probabilité : {proba:.1%}")
                else:
                    st.error(f"❌ Satisfaction insuffisante prévue — probabilité : {proba:.1%}")
            except Exception as e:
                st.error(f"Erreur : {e}")

if __name__ == "__main__":
    build_app()