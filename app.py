import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Prédiction Résultats F1", layout="wide")

@st.cache_resource
def load_models():
    models = {}
    def try_load(name, path):
        p = Path(path)
        if p.exists():
            models[name] = joblib.load(p)
            return True
        return False

    ok_top10 = try_load("top10", "model_top10_rf.pkl")
    ok_time  = try_load("time",  "model_time_rf.pkl")
    ok_win   = try_load("winner","model_winner_rf.pkl")

    return models, {"top10": ok_top10, "time": ok_time, "winner": ok_win}

MODELS, OK = load_models()

def need(col):
    st.toast(f" Modèle '{col}' manquant")

@st.cache_data
def load_hist(df_path_candidates=("dataset_f1_ml_ready.csv","dataset_f1_recent.csv")):
    for p in df_path_candidates:
        if Path(p).exists():
            try:
                df = pd.read_csv(p)
                needed = {"year","round","gp_name","forename","surname","team","grid"}
                if needed.issubset(df.columns):
                    return df
            except Exception:
                pass
    return None

DF_HIST = load_hist()

#if DF_HIST is not None:
    #st.info(f"Dataset chargé : {len(DF_HIST)} lignes et {DF_HIST.shape[1]} colonnes")
#else:
    #st.warning("Aucun dataset historique trouvé.")

def build_df_input(year, round_, gp_name, forename, surname, team, grid, n_pitstops, avg_pit_ms, is_rain):
    return pd.DataFrame([{
        "year": year,
        "round": round_,
        "gp_name": gp_name,
        "forename": forename,
        "surname": surname,
        "team": team,
        "grid": grid,
        "n_pitstops": n_pitstops,
        "avg_pit_ms": avg_pit_ms,
        "is_rain": bool(is_rain),
    }])

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return (model.predict(X).astype(float)).clip(0, 1)

st.sidebar.header("Entrer les paramètres")
year = st.sidebar.number_input("Année", 2018, 2026, 2024)
round_ = st.sidebar.number_input("Round", 1, 25, 1)
gp_name = st.sidebar.text_input("Grand Prix", "Bahrain Grand Prix")
forename = st.sidebar.text_input("Prénom du pilote", "Max")
surname = st.sidebar.text_input("Nom du pilote", "Verstappen")
team = st.sidebar.text_input("Équipe", "Red Bull")
grid = st.sidebar.number_input("Position sur la grille", 1, 20, 2)
n_pitstops = st.sidebar.number_input("Nombre de pitstops", 0, 10, 2)
avg_pit_ms = st.sidebar.number_input("Durée moyenne pitstop (ms)", 1500, 6000, 2300)
is_rain = st.sidebar.checkbox("Pluie ?", False)

st.title(" Prédiction Résultats F1")

st.subheader("Prédiction un pilote")
df_input = build_df_input(year, round_, gp_name, forename, surname, team, grid, n_pitstops, avg_pit_ms, is_rain)
st.dataframe(df_input, hide_index=True, use_container_width=True)

colA, colB, colC = st.columns([1,1,1])
if colA.button("Prédire"):
    with st.spinner("Prédictions en cours…"):
        
        if not OK["top10"]:
            need("Top10")
            proba_top10 = np.nan
            pred_top10 = np.nan
        else:
            proba_top10 = float(safe_predict_proba(MODELS["top10"], df_input)[0])
            pred_top10 = int(MODELS["top10"].predict(df_input)[0])

        if not OK["time"]:
            need("Temps")
            pred_time_ms = np.nan
        else:
            pred_time_ms = float(MODELS["time"].predict(df_input)[0])
        if OK["winner"]:
            proba_win = float(safe_predict_proba(MODELS["winner"], df_input)[0])
        else:
            proba_win = np.nan

    st.write(f"**Probabilité d'être Top10** : {proba_top10:.1%}" if pd.notna(proba_top10) else "Modèle Top10 indisponible")
    st.write(f"**Prédiction Top10** (1=Oui, 0=Non) : {pred_top10}" if pd.notna(pred_top10) else "")
    st.write(f"**Temps prédit** : {pred_time_ms/1000:.1f} s (~{pred_time_ms/60000:.1f} min)" if pd.notna(pred_time_ms) else "Modèle Temps indisponible")
    if pd.notna(proba_win):
        st.write(f"**Probabilité d’être vainqueur (P1)** : {proba_win:.1%}")

st.divider()
st.subheader("Vainqueur — prédire parmi tous les pilotes")

def propose_grid_from_hist(year_all, round_all, gp_all):
    """Essaie de proposer une grille cohérente depuis l'historique."""
    if DF_HIST is None:
        return None
    try:
        df = DF_HIST[
            (DF_HIST["year"] == int(year_all)) &
            (DF_HIST["round"] == int(round_all)) &
            (DF_HIST["gp_name"].astype(str).str.lower() == str(gp_all).lower())
        ][["forename","surname","team","grid"]].dropna()
        if len(df) == 0:
            df = DF_HIST[DF_HIST["gp_name"].astype(str).str.lower() == str(gp_all).lower()]
            df = df.sort_values(["year","round"]).drop_duplicates(["forename","surname","team"], keep="last")
            df = df[["forename","surname","team","grid"]]
        df = df.copy()
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(20).astype(int).clip(1, 20)
        df = df.sort_values("grid").head(20).reset_index(drop=True)
        return df
    except Exception:
        return None

if "engages" not in st.session_state:
    st.session_state.engages = None

if st.button("Charger les engagés"):
    base = propose_grid_from_hist(year, round_, gp_name)
    if base is None or base.empty:
        base = pd.DataFrame({
            "forename": [f"Driver{i}" for i in range(1,21)],
            "surname":  [f"X{i}" for i in range(1,21)],
            "team":     [f"Team{i%10+1}" for i in range(1,21)],
            "grid":     list(range(1,21)),
        })
    st.session_state.engages = base

if st.session_state.engages is None:
    st.info("Clique **Charger les engagés** pour pré-remplir la grille (tu pourras ensuite éditer).")
else:
    st.write("**Édite si besoin** (noms, équipe, grilles) puis lance la prédiction :")
    engages = st.data_editor(
        st.session_state.engages,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "forename": st.column_config.TextColumn("Prénom"),
            "surname":  st.column_config.TextColumn("Nom"),
            "team":     st.column_config.TextColumn("Équipe"),
            "grid":     st.column_config.NumberColumn("Grille", min_value=1, max_value=20, step=1),
        }
    )

    col1, col2 = st.columns([1,3])
    with col1:
        go = st.button("Prédire vainqueur", type="primary")

    if go:
        if not OK["winner"] and not OK["top10"]:
            st.error("Aucun modèle de classement dispo (winner/top10). Entraîne au moins l’un des deux.")
        else:
            rows = []
            for _, r in engages.iterrows():
                X = build_df_input(
                    year, round_, gp_name,
                    str(r.get("forename","")), str(r.get("surname","")), str(r.get("team","")),
                    int(r.get("grid",20)),
                    int(n_pitstops), float(avg_pit_ms), bool(is_rain)
                )
                if OK["winner"]:
                    p_win = float(safe_predict_proba(MODELS["winner"], X)[0])
                else:
                    p10 = float(safe_predict_proba(MODELS["top10"], X)[0]) if OK["top10"] else 0.0
                    tms = float(MODELS["time"].predict(X)[0]) if OK["time"] else 1e9
                    p_win = (p10 ** 2) * (1.0 / (1.0 + (tms / 600000.0)))
                t_pred = float(MODELS["time"].predict(X)[0]) if OK["time"] else np.nan

                rows.append({
                    "forename": r.get("forename",""),
                    "surname":  r.get("surname",""),
                    "team":     r.get("team",""),
                    "grid":     int(r.get("grid",20)),
                    "proba_win": p_win,
                    "time_ms_pred": t_pred
                })

            tab = pd.DataFrame(rows)
            tab = tab.sort_values(["proba_win","grid"], ascending=[False, True]).reset_index(drop=True)

            st.success(f"Vainqueur probable : **{tab.loc[0,'forename']} {tab.loc[0,'surname']}** ({tab.loc[0,'team']}) — {tab.loc[0,'proba_win']:.1%}")
            st.dataframe(
                tab.assign(Proba=lambda d: (d["proba_win"]*100).round(1)).drop(columns=["proba_win"]).rename(columns={"time_ms_pred":"time_ms"}),
                use_container_width=True
            )
            try:
                import matplotlib.pyplot as plt
                top10 = tab.head(10).copy()
                labels = top10["forename"] + " " + top10["surname"]
                fig, ax = plt.subplots(figsize=(8,4))
                ax.barh(labels, top10["proba_win"])
                ax.invert_yaxis()
                ax.set_xlabel("Proba de victoire")
                ax.set_title("Top 10 — Proba de victoire")
                st.pyplot(fig)
            except Exception:
                st.info("Graphique indisponible (matplotlib non trouvé).")



