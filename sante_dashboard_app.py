import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Soins de Santé Primaire – Maroc 2024",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #f0f4f8; }
  [data-testid="stSidebar"] { background: #1a3a6b; }
  [data-testid="stSidebar"] * { color: #e8eef7 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #a8c4e0 !important; font-size:0.82rem; }
  [data-testid="stSidebar"] hr { border-color: #2d5a9e44; }

  /* KPI cards */
  .kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(26,58,107,0.10);
    border-left: 5px solid #1a6bc4;
    height: 110px;
    display: flex; flex-direction: column; justify-content: center;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; color: #1a3a6b; line-height:1.1; }
  .kpi-label { font-size: 0.82rem; color: #6b7fa3; margin-top: 4px; }
  .kpi-delta { font-size: 0.78rem; color: #2ecc71; margin-top:2px; }

  /* Section headers */
  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #1a3a6b;
    border-bottom: 2px solid #1a6bc4; padding-bottom: 6px;
    margin: 24px 0 16px 0;
  }

  /* Chart containers */
  .chart-box {
    background: white; border-radius: 12px;
    padding: 16px; box-shadow: 0 2px 8px rgba(26,58,107,0.08);
  }

  /* Insight cards */
  .insight-card {
    background: linear-gradient(135deg,#eaf2ff,#f5f9ff);
    border-radius: 10px; padding:14px 18px;
    border: 1px solid #b8d4f0; margin-bottom: 10px;
  }
  .insight-icon { font-size:1.3rem; }
  .insight-text { color: #1a3a6b; font-size:0.88rem; }

  /* ML prediction card */
  .pred-result {
    background: linear-gradient(135deg,#1a6bc4,#1a3a6b);
    color: white; border-radius: 12px; padding:20px;
    text-align: center;
  }
  .pred-label { font-size:0.9rem; opacity:0.8; }
  .pred-value { font-size:1.8rem; font-weight:700; }

  /* Sidebar logo area */
  .sidebar-header { text-align:center; padding:10px 0 20px 0; }
  .sidebar-header h2 { color:#a8d4f0 !important; font-size:1.05rem; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: white; border-radius: 8px 8px 0 0;
    padding: 8px 18px; color: #1a3a6b; font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: #1a6bc4 !important; color: white !important;
  }

  /* Remove Streamlit default padding */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  h1 { color: #1a3a6b; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("data.xlsx", header=3)
    df.columns = ["Region", "Delegation", "Commune", "Nom", "Categorie", "_drop", "_abbr", "_desc"]
    df = df.drop(columns=["_drop", "_abbr", "_desc"])
    df = df.dropna(subset=["Region", "Delegation", "Categorie"])
    df["Categorie"] = df["Categorie"].str.strip()
    df["Region"] = df["Region"].str.strip()
    df["Delegation"] = df["Delegation"].str.strip()
    df["Commune"] = df["Commune"].str.strip()

    cat_labels = {
        "CSR-1": "CS Rural Niv.1",
        "CSR-2": "CS Rural Niv.2",
        "CSU-1": "CS Urbain Niv.1",
        "CSU-2": "CS Urbain Niv.2",
        "DR":    "Dispensaire Rural",
        "CDTMR":"Centre Diag. Maladies Resp.",
        "CRSR": "Centre Réf. Santé Reprod.",
        "LSP":  "Laboratoire Santé Pub.",
    }
    df["Categorie_Label"] = df["Categorie"].map(cat_labels).fillna(df["Categorie"])

    # Rural vs Urban flag
    df["Milieu"] = df["Categorie"].apply(
        lambda x: "Rural" if x in ["CSR-1", "CSR-2", "DR"] else
                  ("Urbain" if x in ["CSU-1", "CSU-2"] else "Spécialisé")
    )
    return df

df_all = load_data()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2>🏥 Tableau de Bord<br>Santé Primaire Maroc 2024</h2></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🔍 Filtres Interactifs**")

    all_regions = ["Toutes les Régions"] + sorted(df_all["Region"].unique().tolist())
    sel_region = st.selectbox("Région", all_regions)

    if sel_region == "Toutes les Régions":
        deleg_options = ["Toutes"] + sorted(df_all["Delegation"].unique().tolist())
    else:
        deleg_options = ["Toutes"] + sorted(df_all[df_all["Region"] == sel_region]["Delegation"].unique().tolist())
    sel_delegation = st.selectbox("Délégation", deleg_options)

    all_cats = df_all["Categorie"].unique().tolist()
    sel_cats = st.multiselect("Catégorie d'établissement", sorted(all_cats), default=sorted(all_cats))

    st.markdown("---")
    st.markdown("**ℹ️ À propos**")
    st.markdown("<span style='font-size:0.78rem;color:#a8c4e0'>Source : Ministère de la Santé et de la Protection Sociale – Données 2024</span>", unsafe_allow_html=True)

# ─── FILTER DATA ──────────────────────────────────────────────────────────────
df = df_all.copy()
if sel_region != "Toutes les Régions":
    df = df[df["Region"] == sel_region]
if sel_delegation != "Toutes":
    df = df[df["Delegation"] == sel_delegation]
if sel_cats:
    df = df[df["Categorie"].isin(sel_cats)]

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 🏥 Établissements de Soins de Santé Primaire – Maroc 2024")
st.markdown(f"<span style='color:#6b7fa3;font-size:0.9rem'>Vue filtrée : <b>{len(df):,}</b> établissements · <b>{df['Region'].nunique()}</b> région(s) · <b>{df['Delegation'].nunique()}</b> délégation(s)</span>", unsafe_allow_html=True)

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
cards = [
    (k1, str(len(df)), "Établissements", "#1a6bc4"),
    (k2, str(df["Region"].nunique()), "Régions", "#0e9e6b"),
    (k3, str(df["Delegation"].nunique()), "Délégations", "#7c3aed"),
    (k4, str(df["Commune"].nunique()), "Communes", "#dc6027"),
    (k5, str(df["Categorie"].nunique()), "Catégories", "#1a9ec4"),
]
for col, val, lbl, color in cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:{color}">
          <div class="kpi-value" style="color:{color}">{val}</div>
          <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'Ensemble", "🔬 Analyse Approfondie", "🤖 Machine Learning", "💡 Insights & Recommandations"])

PALETTE = ["#1a6bc4","#0e9e6b","#7c3aed","#dc6027","#1a9ec4","#e84393","#f59e0b","#64748b"]

# ══════════ TAB 1 – OVERVIEW ══════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="section-title">Répartition par Région</div>', unsafe_allow_html=True)
        reg_count = df.groupby("Region").size().reset_index(name="Total").sort_values("Total", ascending=True)
        fig = px.bar(
            reg_count, x="Total", y="Region", orientation="h",
            color="Total", color_continuous_scale=["#a8d4f0","#1a6bc4","#1a3a6b"],
            text="Total", labels={"Total":"Nombre","Region":"Région"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=420, showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis_title="", margin=dict(l=0,r=30,t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Par Catégorie</div>', unsafe_allow_html=True)
        cat_count = df.groupby("Categorie_Label").size().reset_index(name="Total").sort_values("Total", ascending=False)
        fig2 = px.pie(
            cat_count, values="Total", names="Categorie_Label",
            color_discrete_sequence=PALETTE, hole=0.42
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        fig2.update_layout(
            height=420, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap region × category
    st.markdown('<div class="section-title">Heatmap Régions × Catégories</div>', unsafe_allow_html=True)
    heat = df.groupby(["Region","Categorie"]).size().unstack(fill_value=0)
    fig3 = px.imshow(
        heat, color_continuous_scale="Blues",
        labels=dict(x="Catégorie", y="Région", color="Nbre"),
        aspect="auto", text_auto=True
    )
    fig3.update_layout(
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=10,b=10),
        xaxis_title="", yaxis_title=""
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Rural vs Urban
    st.markdown('<div class="section-title">Milieu de Couverture (Rural / Urbain / Spécialisé)</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        milieu_reg = df.groupby(["Region","Milieu"]).size().reset_index(name="N")
        fig4 = px.bar(milieu_reg, x="Region", y="N", color="Milieu",
                      color_discrete_map={"Rural":"#1a6bc4","Urbain":"#0e9e6b","Spécialisé":"#dc6027"},
                      barmode="stack", labels={"N":"Nbre","Region":""})
        fig4.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_tickangle=-35, legend=dict(orientation="h", y=1.1),
                           margin=dict(l=0,r=0,t=30,b=60))
        st.plotly_chart(fig4, use_container_width=True)
    with m2:
        milieu_tot = df.groupby("Milieu").size().reset_index(name="N")
        fig5 = px.bar(milieu_tot, x="Milieu", y="N", color="Milieu",
                      color_discrete_map={"Rural":"#1a6bc4","Urbain":"#0e9e6b","Spécialisé":"#dc6027"},
                      text="N", labels={"N":"Nbre","Milieu":""})
        fig5.update_traces(textposition="outside")
        fig5.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                           showlegend=False, margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig5, use_container_width=True)

# ══════════ TAB 2 – DEEP ANALYSIS ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Top 20 Délégations par Nombre d\'Établissements</div>', unsafe_allow_html=True)
    top_del = df.groupby("Delegation").size().reset_index(name="Total").nlargest(20,"Total")
    fig6 = px.bar(
        top_del, x="Delegation", y="Total",
        color="Total", color_continuous_scale=["#a8d4f0","#1a6bc4","#1a3a6b"],
        text="Total", labels={"Total":"Nbre","Delegation":"Délégation"}
    )
    fig6.update_traces(textposition="outside")
    fig6.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                       plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_tickangle=-40, margin=dict(l=0,r=0,t=10,b=80))
    st.plotly_chart(fig6, use_container_width=True)

    # Stacked bar delegation × category
    if sel_region != "Toutes les Régions" or sel_delegation != "Toutes":
        st.markdown('<div class="section-title">Composition des Délégations par Catégorie</div>', unsafe_allow_html=True)
        del_cat = df.groupby(["Delegation","Categorie"]).size().reset_index(name="N")
        fig7 = px.bar(del_cat, x="Delegation", y="N", color="Categorie",
                      barmode="stack", color_discrete_sequence=PALETTE,
                      labels={"N":"Nbre","Delegation":"","Categorie":"Catégorie"})
        fig7.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_tickangle=-35, margin=dict(l=0,r=0,t=10,b=80),
                           legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig7, use_container_width=True)

    # Coverage ratio analysis
    st.markdown('<div class="section-title">Taux de Couverture Rurale vs Urbaine par Région</div>', unsafe_allow_html=True)
    cov = df_all.groupby(["Region","Milieu"]).size().unstack(fill_value=0).reset_index()
    for col in ["Rural","Urbain","Spécialisé"]:
        if col not in cov.columns:
            cov[col] = 0
    cov["Total"] = cov["Rural"] + cov["Urbain"] + cov["Spécialisé"]
    cov["Ratio_Rural"] = (cov["Rural"] / cov["Total"] * 100).round(1)
    cov["Ratio_Urbain"] = (cov["Urbain"] / cov["Total"] * 100).round(1)
    cov_sorted = cov.sort_values("Ratio_Rural", ascending=True)

    fig8 = go.Figure()
    fig8.add_trace(go.Bar(name="Rural", x=cov_sorted["Ratio_Rural"], y=cov_sorted["Region"],
                          orientation="h", marker_color="#1a6bc4", text=cov_sorted["Ratio_Rural"].astype(str)+"%",
                          textposition="inside"))
    fig8.add_trace(go.Bar(name="Urbain", x=cov_sorted["Ratio_Urbain"], y=cov_sorted["Region"],
                          orientation="h", marker_color="#0e9e6b", text=cov_sorted["Ratio_Urbain"].astype(str)+"%",
                          textposition="inside"))
    fig8.update_layout(barmode="stack", height=400, plot_bgcolor="white", paper_bgcolor="white",
                       xaxis=dict(title="% des établissements", range=[0,100]),
                       yaxis_title="", margin=dict(l=0,r=0,t=10,b=10),
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig8, use_container_width=True)

    # Treemap
    st.markdown('<div class="section-title">Carte Hiérarchique (Région → Délégation → Catégorie)</div>', unsafe_allow_html=True)
    treemap_df = df.copy()
    treemap_df["count"] = 1
    fig9 = px.treemap(
        treemap_df, path=["Region","Delegation","Categorie"], values="count",
        color="Categorie", color_discrete_sequence=PALETTE
    )
    fig9.update_layout(height=500, margin=dict(l=0,r=0,t=10,b=10))
    st.plotly_chart(fig9, use_container_width=True)

# ══════════ TAB 3 – MACHINE LEARNING ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🤖 Composante Machine Learning</div>', unsafe_allow_html=True)

    ml_tab1, ml_tab2 = st.tabs(["Clustering des Régions", "Prédiction de Catégorie"])

    # ── ML Tab 1 : K-Means Clustering ────────────────────────────────────────
    with ml_tab1:
        st.markdown("""
        **Objectif :** Regrouper les régions selon leur profil de couverture sanitaire à l'aide du K-Means clustering.
        Cela permet d'identifier des régions similaires et des inégalités structurelles.
        """)

        # Build feature matrix
        feat = df_all.groupby(["Region","Categorie"]).size().unstack(fill_value=0)
        feat["Total"] = feat.sum(axis=1)
        for col in feat.columns[:-1]:
            feat[f"Pct_{col}"] = (feat[col] / feat["Total"] * 100).round(2)
        pct_cols = [c for c in feat.columns if c.startswith("Pct_")]
        X = feat[pct_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = st.slider("Nombre de clusters", 2, 6, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        feat["Cluster"] = kmeans.fit_predict(X_scaled)
        feat["Cluster"] = feat["Cluster"].astype(str)
        feat_reset = feat.reset_index()

        # PCA for 2D viz
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        feat_reset["PC1"] = coords[:, 0]
        feat_reset["PC2"] = coords[:, 1]

        c_left, c_right = st.columns([2,1])
        with c_left:
            fig_cl = px.scatter(
                feat_reset, x="PC1", y="PC2", color="Cluster",
                text="Region", size="Total",
                color_discrete_sequence=PALETTE,
                labels={"PC1":"Composante Principale 1","PC2":"Composante Principale 2"},
                title="Clustering PCA – Profils Régionaux de Couverture Sanitaire"
            )
            fig_cl.update_traces(textposition="top center")
            fig_cl.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white",
                                 margin=dict(l=0,r=0,t=40,b=10))
            st.plotly_chart(fig_cl, use_container_width=True)

        with c_right:
            st.markdown("**Composition des clusters :**")
            for cluster_id in sorted(feat_reset["Cluster"].unique()):
                members = feat_reset[feat_reset["Cluster"]==cluster_id]["Region"].tolist()
                color = PALETTE[int(cluster_id)]
                st.markdown(f"""
                <div style="background:white;border-left:5px solid {color};
                  border-radius:8px;padding:10px 14px;margin-bottom:8px;
                  box-shadow:0 2px 6px rgba(0,0,0,0.07)">
                  <b style="color:{color}">Cluster {cluster_id}</b><br>
                  <span style="font-size:0.82rem;color:#444">{'<br>'.join(members)}</span>
                </div>""", unsafe_allow_html=True)

        # Radar chart for cluster centers
        st.markdown("**Profil moyen par cluster (% par catégorie) :**")
        cluster_profiles = feat_reset.groupby("Cluster")[pct_cols].mean().round(1)
        short_labels = [c.replace("Pct_","") for c in pct_cols]
        fig_radar = go.Figure()
        for i, row in cluster_profiles.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=short_labels + [short_labels[0]],
                fill="toself", name=f"Cluster {i}",
                line_color=PALETTE[int(i)]
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, cluster_profiles.values.max()+5])),
            showlegend=True, height=420, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=40,r=40,t=40,b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── ML Tab 2 : Prediction ────────────────────────────────────────────────
    with ml_tab2:
        st.markdown("""
        **Objectif :** Prédire la catégorie d'un nouvel établissement sanitaire à partir des caractéristiques 
        de sa localisation (région et délégation). Utile pour l'**aide à la planification** sanitaire.
        """)

        # Encode features
        enc_df = df_all[["Region","Delegation","Categorie"]].copy()
        enc_df["Region_enc"] = pd.Categorical(enc_df["Region"]).codes
        enc_df["Delegation_enc"] = pd.Categorical(enc_df["Delegation"]).codes
        region_map = dict(zip(enc_df["Region"], enc_df["Region_enc"]))
        deleg_map  = dict(zip(enc_df["Delegation"], enc_df["Delegation_enc"]))

        # Filter rare classes
        class_counts = enc_df["Categorie"].value_counts()
        valid_classes = class_counts[class_counts >= 10].index
        enc_df = enc_df[enc_df["Categorie"].isin(valid_classes)]

        X_ml = enc_df[["Region_enc","Delegation_enc"]].values
        y_ml = enc_df["Categorie"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml)

        rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        rf.fit(X_tr, y_tr)
        acc = rf.score(X_te, y_te)

        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:16px;
          box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:20px;display:flex;gap:40px">
          <div>
            <div style="font-size:0.8rem;color:#6b7fa3">Précision du Modèle</div>
            <div style="font-size:2rem;font-weight:700;color:#1a6bc4">{acc*100:.1f}%</div>
          </div>
          <div>
            <div style="font-size:0.8rem;color:#6b7fa3">Algorithme</div>
            <div style="font-size:1.1rem;font-weight:600;color:#1a3a6b">Random Forest</div>
          </div>
          <div>
            <div style="font-size:0.8rem;color:#6b7fa3">Données d'entraînement</div>
            <div style="font-size:1.1rem;font-weight:600;color:#1a3a6b">{len(X_tr):,} établissements</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Prediction form
        st.markdown("### 🔮 Prédire pour un Nouvel Établissement")
        col_r, col_d = st.columns(2)
        with col_r:
            pred_region = st.selectbox("Région de l'établissement", sorted(df_all["Region"].unique()))
        with col_d:
            deleg_opts = sorted(df_all[df_all["Region"]==pred_region]["Delegation"].unique())
            pred_deleg = st.selectbox("Délégation", deleg_opts)

        if st.button("🔍 Lancer la Prédiction", use_container_width=True):
            r_enc = region_map.get(pred_region, 0)
            d_enc = deleg_map.get(pred_deleg, 0)
            pred_proba = rf.predict_proba([[r_enc, d_enc]])[0]
            classes   = rf.classes_

            cat_labels = {
                "CSR-1":"CS Rural Niv.1","CSR-2":"CS Rural Niv.2",
                "CSU-1":"CS Urbain Niv.1","CSU-2":"CS Urbain Niv.2",
                "DR":"Dispensaire Rural","CDTMR":"Diag. Maladies Resp.",
                "CRSR":"Réf. Santé Reprod.","LSP":"Labo. Santé Pub."
            }
            top_class = classes[np.argmax(pred_proba)]
            top_prob  = pred_proba.max()

            res_col, dist_col = st.columns([1, 2])
            with res_col:
                st.markdown(f"""
                <div class="pred-result">
                  <div class="pred-label">Catégorie Prédite</div>
                  <div class="pred-value">{top_class}</div>
                  <div class="pred-label" style="margin-top:6px">{cat_labels.get(top_class,'')} · Confiance : {top_prob*100:.0f}%</div>
                </div>""", unsafe_allow_html=True)
            with dist_col:
                prob_df = pd.DataFrame({"Catégorie": classes, "Probabilité (%)": (pred_proba*100).round(1)})
                prob_df = prob_df.sort_values("Probabilité (%)", ascending=True)
                fig_bar = px.bar(prob_df, x="Probabilité (%)", y="Catégorie", orientation="h",
                                 color="Probabilité (%)", color_continuous_scale=["#d4e9ff","#1a6bc4"],
                                 text="Probabilité (%)", labels={"Catégorie":""})
                fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
                fig_bar.update_layout(height=280, showlegend=False, coloraxis_showscale=False,
                                      plot_bgcolor="white", paper_bgcolor="white",
                                      margin=dict(l=0,r=30,t=10,b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        # Feature importance
        st.markdown("**Importance des Variables (Random Forest) :**")
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"Variable":["Région","Délégation"],"Importance":importances})
        fig_imp = px.bar(imp_df, x="Variable", y="Importance",
                         color="Importance", color_continuous_scale=["#a8d4f0","#1a6bc4"],
                         text=imp_df["Importance"].round(3))
        fig_imp.update_traces(textposition="outside")
        fig_imp.update_layout(height=280, showlegend=False, coloraxis_showscale=False,
                              plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

# ══════════ TAB 4 – INSIGHTS ══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">💡 Insights Automatiques & Recommandations</div>', unsafe_allow_html=True)

    # Compute insights from full dataset
    reg_totals = df_all.groupby("Region").size()
    top_region = reg_totals.idxmax()
    bot_region = reg_totals.idxmin()
    pct_rural  = (df_all[df_all["Milieu"]=="Rural"].shape[0] / len(df_all) * 100)
    pct_urban  = (df_all[df_all["Milieu"]=="Urbain"].shape[0] / len(df_all) * 100)
    pct_spec   = (df_all[df_all["Milieu"]=="Spécialisé"].shape[0] / len(df_all) * 100)

    # Regions with no LSP
    lsp_regions = df_all[df_all["Categorie"]=="LSP"]["Region"].unique()
    no_lsp = [r for r in df_all["Region"].unique() if r not in lsp_regions]

    # Region with highest rural ratio
    rr = df_all.groupby(["Region","Milieu"]).size().unstack(fill_value=0)
    rr["Total"] = rr.sum(axis=1)
    rr["Rural_pct"] = rr.get("Rural", 0) / rr["Total"] * 100
    most_rural_reg = rr["Rural_pct"].idxmax()
    most_rural_pct = rr["Rural_pct"].max()

    insights = [
        ("🏆", f"<b>{top_region}</b> est la région la plus couverte avec <b>{reg_totals[top_region]:,}</b> établissements."),
        ("⚠️", f"<b>{bot_region}</b> est la région la moins dotée avec seulement <b>{reg_totals[bot_region]}</b> établissements — un potentiel déficit de couverture."),
        ("🌾", f"La couverture sanitaire est principalement <b>rurale</b> : <b>{pct_rural:.1f}%</b> des établissements sont ruraux (DR + CSR), contre <b>{pct_urban:.1f}%</b> urbains."),
        ("🔬", f"Les établissements spécialisés (CDTMR, CRSR, LSP) ne représentent que <b>{pct_spec:.1f}%</b> du réseau — des ressources critiques pour des pathologies complexes."),
        ("🏥", f"<b>{most_rural_reg}</b> présente le taux de ruralité le plus élevé : <b>{most_rural_pct:.1f}%</b> d'établissements ruraux."),
        ("🧪", f"Les Laboratoires de Santé Publique (LSP) sont absents dans : <b>{', '.join(no_lsp) if no_lsp else 'toutes les régions en ont'}</b>."),
        ("📊", f"Le réseau national compte <b>{len(df_all):,}</b> établissements répartis dans <b>{df_all['Region'].nunique()}</b> régions et <b>{df_all['Delegation'].nunique()}</b> délégations."),
        ("🎯", f"Les CS Rural Niv.1 (CSR-1) constituent la colonne vertébrale du réseau avec <b>{(df_all['Categorie']=='CSR-1').sum():,}</b> unités."),
    ]

    for icon, text in insights:
        st.markdown(f"""
        <div class="insight-card">
          <span class="insight-icon">{icon}</span>&nbsp;&nbsp;<span class="insight-text">{text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Recommandations Stratégiques</div>', unsafe_allow_html=True)

    recs = [
        ("🔴 Priorité Haute", f"Renforcer la couverture à <b>{bot_region}</b> : augmenter le nombre d'établissements de premier recours (CSR-1, CSU-1).", "#fee2e2", "#dc2626"),
        ("🟡 Priorité Moyenne", "Étendre les laboratoires de santé publique (LSP) dans les régions qui en sont dépourvues pour améliorer la capacité diagnostique.", "#fef9c3", "#ca8a04"),
        ("🟢 Bonne Pratique", f"Capitaliser sur le modèle de <b>{top_region}</b> comme exemple de réseau bien développé pour guider d'autres régions.", "#dcfce7", "#16a34a"),
        ("🔵 Innovation", "Déployer des centres mobiles de santé dans les zones rurales éloignées, notamment dans les régions à fort taux de ruralité.", "#dbeafe", "#1d4ed8"),
        ("🟣 Qualité", "Upgrader progressivement les Dispensaires Ruraux (DR) vers des CSR-1 ou CSR-2 pour améliorer le niveau de soins disponibles.", "#ede9fe", "#7c3aed"),
    ]

    for title, text, bg, border in recs:
        st.markdown(f"""
        <div style="background:{bg};border-left:5px solid {border};border-radius:10px;
          padding:14px 18px;margin-bottom:10px">
          <b style="color:{border}">{title}</b><br>
          <span style="font-size:0.88rem;color:#1a3a6b">{text}</span>
        </div>""", unsafe_allow_html=True)

    # Data table
    st.markdown('<div class="section-title">📄 Données Filtrées</div>', unsafe_allow_html=True)
    st.dataframe(df[["Region","Delegation","Commune","Nom","Categorie","Milieu"]].reset_index(drop=True),
                 use_container_width=True, height=350)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exporter les données filtrées (CSV)", csv, "donnees_sante_filtrees.csv", "text/csv")
