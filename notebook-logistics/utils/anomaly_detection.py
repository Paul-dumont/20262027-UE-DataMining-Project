from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

VALEURS_VIDES = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


# ======================================================================
# 1. Détection de la nature des colonnes (et conversion en nombres / dates)
# ======================================================================
def _texte_propre(serie: pd.Series) -> pd.Series:
    t = serie.astype("string").str.strip()
    return t.mask(t.str.lower().isin(VALEURS_VIDES))


def _convertir_nombres(serie: pd.Series, seuil: float = 0.9) -> Optional[pd.Series]:
    """Texte -> nombres si >= seuil des valeurs sont numériques ("2469,48", "88 case", "1 200")."""
    t = _texte_propre(serie)
    valides = t.dropna()
    if valides.empty or valides.str.match(r"^0\d+$").any():  # codes du type "007"
        return None
    sans_espace = t.str.replace(r"[\s\u00a0](?=\d{3}\b)", "", regex=True)
    sans_espace = sans_espace.str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True)
    nombres = pd.to_numeric(sans_espace, errors="coerce")
    # "88 case", "1 service" : un nombre suivi d'une unité (lettres uniquement)
    motif = r"^\s*(-?\d+(?:[.,]\d+)?)\s+[A-Za-z_\-]+\s*$"
    avec_unite = t.str.extract(motif)[0].str.replace(",", ".", regex=False)
    nombres = nombres.fillna(pd.to_numeric(avec_unite, errors="coerce")).astype(float)
    return nombres if nombres.notna().sum() / len(valides) >= seuil else None


def _convertir_dates(serie: pd.Series, seuil: float = 0.8) -> Optional[pd.Series]:
    """Texte -> dates (formats mixtes acceptés) si >= seuil des valeurs sont reconnues."""
    t = _texte_propre(serie)
    valides = t.dropna()
    if valides.empty:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(t, format="mixed", errors="coerce")
    return dates if dates.notna().sum() / len(valides) >= seuil else None


def _nature_colonne(serie: pd.Series):
    """Renvoie (nature, serie_convertie) ; nature : numérique, date, catégorielle, identifiant, vide."""
    if serie.dropna().empty:
        return "vide", serie
    if pd.api.types.is_bool_dtype(serie):
        return "catégorielle", serie
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "date", serie
    if pd.api.types.is_numeric_dtype(serie):
        s = serie.astype(float)
        n = int(s.notna().sum())
        entiers = bool((s.dropna() % 1 == 0).all())
        if entiers and n > 20 and s.nunique() / n > 0.98 and s.is_monotonic_increasing:
            return "identifiant", s          # numéro de ligne, index...
        return "numérique", s
    nombres = _convertir_nombres(serie)
    if nombres is not None:
        return "numérique", nombres
    dates = _convertir_dates(serie)
    if dates is not None:
        return "date", dates
    s = serie.astype("object")
    n = int(s.notna().sum())
    if n > 5 and s.nunique() / n > 0.9:
        return "identifiant", s
    return "catégorielle", s


def _jours(dates: pd.Series) -> pd.Series:
    return (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)


# ======================================================================
# 2. Préparation : table numérique propre
# ======================================================================
def preparer_numeriques(
    df: pd.DataFrame,
    colonnes: Optional[Iterable[str]] = None,
    inclure_dates: bool = False,
    max_manquants: float = 0.3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extrait les variables exploitables en DataFrame de float (même index que df).
    Exclut : catégories, identifiants, colonnes constantes ou trop vides (> max_manquants).
    Si `colonnes` est fourni, seules ces colonnes sont examinées.
    """
    cols = [c for c in (colonnes if colonnes is not None else df.columns) if c in df.columns]
    inconnues = [c for c in (colonnes or []) if c not in df.columns]
    if inconnues:
        print(f"Colonnes introuvables (ignorées) : {inconnues}")
    data, exclues = {}, []
    for c in cols:
        nature, s = _nature_colonne(df[c])
        if nature == "date" and inclure_dates:
            s, nature = _jours(s), "numérique"
        if nature != "numérique":
            exclues.append(f"{c} ({nature})")
            continue
        s = pd.Series(s.to_numpy(dtype=float), index=df.index)
        if s.isna().mean() > max_manquants:
            exclues.append(f"{c} ({s.isna().mean():.0%} de manquants)")
        elif s.nunique(dropna=True) < 2:
            exclues.append(f"{c} (constante)")
        else:
            data[c] = s
    if verbose:
        print(f"Variables utilisées ({len(data)}) : {list(data)}")
        if exclues:
            print(f"Variables exclues : {exclues}")
    return pd.DataFrame(data, index=df.index)


# ======================================================================
# 3. Univariée : IQR
# ======================================================================
def bornes_iqr(s: pd.Series, k: float = 1.5):
    """(Q1, Q3, IQR, borne basse, borne haute) de la règle de Tukey."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1, q3, iqr, q1 - k * iqr, q3 + k * iqr


def _hors_bornes(s: pd.Series, k: float = 1.5) -> pd.Series:
    _, _, iqr, bas, haut = bornes_iqr(s.dropna(), k) if s.notna().any() else (0, 0, 0, np.nan, np.nan)
    return ((s < bas) | (s > haut)).fillna(False)


def detection_iqr(
    table: pd.DataFrame,
    groupes: Optional[pd.Series] = None,
    k: float = 1.5,
    afficher: bool = True,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    IQR colonne par colonne. Si `groupes` est fourni, un second calcul est fait par groupe.
    Renvoie (indicateurs, résumé) :
      - indicateurs : DataFrame de booléens (True = hors de [Q1-k·IQR ; Q3+k·IQR]) ;
        si `groupes` est donné, l'IQR PAR GROUPE est utilisé, sinon l'IQR global.
      - résumé : bornes et nombre de valeurs atypiques (global, et par groupe si applicable).
    """
    flags = pd.DataFrame(False, index=table.index, columns=table.columns)
    lignes = []
    for c in table.columns:
        s = table[c]
        q1, q3, iqr, bas, haut = bornes_iqr(s.dropna(), k)
        glob = _hors_bornes(s, k)
        ligne = {"variable": c, "Q1": q1, "Q3": q3, "IQR": iqr, "borne basse": bas,
                 "borne haute": haut, "atypiques (IQR global)": int(glob.sum())}
        if groupes is not None:
            par_groupe = s.groupby(groupes, dropna=False).transform(lambda x: _hors_bornes(x, k))
            flags[c] = par_groupe.astype(bool)
            ligne["atypiques (IQR par groupe)"] = int(flags[c].sum())
        else:
            flags[c] = glob
        lignes.append(ligne)
    resume = pd.DataFrame(lignes).set_index("variable")

    if afficher and len(table.columns):
        n = len(table.columns)
        cols_fig = min(n, 4)
        lig_fig = int(np.ceil(n / cols_fig))
        fig, axes = plt.subplots(lig_fig, cols_fig, figsize=figsize or (4 * cols_fig, 4.2 * lig_fig),
                                 squeeze=False)
        for ax, c in zip(axes.ravel(), table.columns):
            ax.boxplot(table[c].dropna(), whis=1.5, flierprops=dict(marker="o", markersize=3, alpha=.5))
            haut, bas = resume.loc[c, "borne haute"], resume.loc[c, "borne basse"]
            if haut < table[c].max():
                ax.axhline(haut, color="crimson", ls="--", lw=1, label=f"borne haute = {haut:,.4g}")
            if bas > table[c].min():
                ax.axhline(bas, color="crimson", ls="--", lw=1, label=f"borne basse = {bas:,.4g}")
            ax.set_title(c, fontsize=10)
            ax.set_xticks([])
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="upper right")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle(f"Détection univariée : boîtes à moustaches (IQR global, k = {k})")
        fig.tight_layout()
        _finir(fig, save_path, show)
    return flags, resume


# ======================================================================
# 4. Matrice pour la détection multivariée
# ======================================================================
def matrice_multivariee(
    table: pd.DataFrame,
    groupes: Optional[pd.Series] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Prépare la matrice Z (centrée-réduite) des méthodes multivariées.
    - Avec `groupes` : chaque variable est exprimée par rapport à la médiane de son groupe
      (log du rapport si toutes les valeurs sont > 0, sinon simple différence).
    - Valeurs manquantes remplacées par la médiane (signalé).
    - Variables forcées à une échelle comparable (StandardScaler) : indispensable
      pour toute méthode fondée sur les distances ou les angles.
    """
    X = table.copy()
    if groupes is not None:
        for c in X.columns:
            med = X[c].groupby(groupes, dropna=False).transform("median")
            if (X[c].dropna() > 0).all():
                X[c] = np.log(X[c] / med)
            else:
                X[c] = X[c] - med
        X = X.loc[:, X.nunique(dropna=True) > 1]    # une variable devenue constante n'apporte rien
        if verbose:
            print("Variables exprimées relativement à la médiane de leur groupe (log du ratio si > 0).")
    n_na = int(X.isna().sum().sum())
    if n_na and verbose:
        print(f"{n_na} valeurs manquantes remplacées par la médiane de la colonne.")
    X = X.fillna(X.median())
    return StandardScaler().fit_transform(X)


# ======================================================================
# 5. Multivariée : clustering (K-means + DBSCAN)
# ======================================================================
def _eps_auto(Z: np.ndarray, min_samples: int, contamination: float) -> float:
    """
    eps de DBSCAN = quantile (1 - 2·contamination) des distances au min_samples-ième voisin.
    Un point n'est « bruit » que s'il est loin de tout point dense : le bruit reste donc
    au plus de l'ordre de `contamination` des lignes (le coude de la courbe est trop instable
    quand il y a beaucoup de valeurs identiques).
    """
    d = NearestNeighbors(n_neighbors=min_samples + 1).fit(Z).kneighbors(Z)[0][:, -1]
    return float(np.quantile(d, 1 - 2 * contamination))


def detection_clustering(
    Z: np.ndarray,
    k_max: int = 8,
    eps: Optional[float] = None,
    min_samples: int = 5,
    k_iqr: float = 1.5,
    contamination: float = 0.03,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """
    K-means : k choisi par le coefficient de silhouette (2..k_max) ; score = distance au centre de
              son groupe ; atypique si > Q3 + k_iqr·IQR de ces distances.
    DBSCAN  : eps estimé automatiquement (quantile des distances aux voisins) si non fourni ; atypiques = points « bruit ».
    """
    n = len(Z)
    res = {}
    k_max = min(k_max, n - 1)
    scores = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=random_state).fit_predict(Z))
              for k in range(2, k_max + 1)}
    k_opt = max(scores, key=scores.get)
    km = KMeans(k_opt, n_init=10, random_state=random_state).fit(Z)
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    seuil = bornes_iqr(pd.Series(dist), k_iqr)[4]
    res.update(cluster_kmeans=km.labels_, dist_centre=dist, out_kmeans=dist > seuil)

    eps_utilise = eps if eps is not None else _eps_auto(Z, min_samples, contamination)
    db = DBSCAN(eps=eps_utilise, min_samples=min_samples).fit(Z)
    res.update(cluster_dbscan=db.labels_, out_dbscan=db.labels_ == -1)

    if verbose:
        print(f"K-means : k = {k_opt} (silhouette {scores[k_opt]:.2f}), seuil de distance = {seuil:.2f} "
              f"-> {int(res['out_kmeans'].sum())} atypiques")
        if scores[k_opt] < 0.35:
            print("   (silhouette faible : peu de groupes nets ; les scores de distance restent exploitables)")
        print(f"DBSCAN  : eps = {eps_utilise:.2f}, min_samples = {min_samples} -> "
              f"{len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)} groupes, "
              f"{int(res['out_dbscan'].sum())} points bruit")
    res.update(k_opt=k_opt, seuil_kmeans=seuil, eps=eps_utilise, silhouettes=scores)
    return res


# ======================================================================
# 6. Angle-based Outlier Detection (FastABOD)
# ======================================================================
def fast_abod(Z: np.ndarray, k: int = 20) -> np.ndarray:
    """
    ABOF de chaque point (FastABOD : paires de ses k plus proches voisins seulement).
    ABOF(A) = variance pondérée de  <AB,AC> / (|AB|²·|AC|²)  avec le poids 1 / (|AB|·|AC|).
    Plus l'ABOF est PETIT, plus le point est atypique (voisins vus sous des angles proches =
    point situé à l'extérieur du nuage).
    """
    n = len(Z)
    k = max(2, min(k, n - 1))
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    iu = np.triu_indices(k, 1)
    abof = np.empty(n)
    for a in range(n):
        V = Z[idx[a]] - Z[a]
        r2 = np.maximum(np.einsum("ij,ij->i", V, V), 1e-12)
        S = (V @ V.T) / (r2[:, None] * r2[None, :])
        W = 1.0 / np.sqrt(r2[:, None] * r2[None, :])
        s, w = S[iu], W[iu]
        m = np.sum(w * s) / np.sum(w)
        abof[a] = np.sum(w * s ** 2) / np.sum(w) - m ** 2
    return abof


def detection_abod(Z: np.ndarray, k: int = 20, contamination: float = 0.03, verbose: bool = True) -> dict:
    abof = fast_abod(Z, k)
    seuil = float(np.quantile(abof, contamination))
    out = abof <= seuil
    if verbose:
        print(f"ABOD    : FastABOD (k = {min(k, len(Z) - 1)}), {contamination:.0%} d'ABOF les plus bas "
              f"(seuil = {seuil:.4g}) -> {int(out.sum())} atypiques")
    return {"abof": abof, "out_abod": out, "seuil_abod": seuil}


# ======================================================================
# 7. Fonction principale
# ======================================================================
@dataclass
class ResultatAnomalies:
    donnees: pd.DataFrame                  # df d'origine + indicateurs / scores
    anomalies: pd.DataFrame                # lignes retenues (>= min_methodes méthodes multivariées)
    resume_iqr: pd.DataFrame
    variables: List[str] = field(default_factory=list)
    parametres: Dict = field(default_factory=dict)


def detecter_anomalies(
    df: pd.DataFrame,
    colonnes: Optional[Iterable[str]] = None,
    group_by: Optional[str] = None,
    inclure_dates: bool = False,
    supprimer_doublons: bool = True,
    k_iqr: float = 1.5,
    contamination: float = 0.03,
    k_abod: int = 20,
    eps: Optional[float] = None,
    min_samples: int = 5,
    min_methodes: int = 2,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> ResultatAnomalies:
    """
    Lance IQR + K-means + DBSCAN + ABOD et les confronte.

    Paramètres principaux
    ---------------------
    colonnes       : colonnes à analyser (défaut : toutes les colonnes numériques détectées).
    group_by       : colonne de regroupement (ex. "item_code") : IQR par groupe et écarts à la
                     médiane du groupe pour la détection multivariée.
    contamination  : part de lignes que l'ABOD doit signaler (0.03 = 3 %).
    min_methodes   : nombre de méthodes multivariées (sur 3) qui doivent s'accorder pour qu'une
                     ligne soit dans `.anomalies`.

    Ajoutées à `.donnees` : iqr_<col> (bool, une colonne par variable), n_iqr, out_kmeans,
    dist_centre, out_dbscan, out_abod, abof, n_methodes, anomalie_consensus.
    """
    if supprimer_doublons:
        n0 = len(df)
        df = df.drop_duplicates()
        if len(df) < n0:
            print(f"{n0 - len(df)} doublon(s) exact(s) supprimé(s).")
    df = df.reset_index(drop=True)
    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"group_by='{group_by}' est introuvable. Colonnes : {list(df.columns)}")
    groupes = df[group_by].astype("string").str.strip() if group_by else None

    colonnes = [c for c in (colonnes if colonnes is not None else df.columns) if c != group_by]
    table = preparer_numeriques(df, colonnes, inclure_dates=inclure_dates)
    if table.shape[1] == 0:
        raise ValueError("Aucune variable numérique exploitable. Précisez `colonnes=` ou `inclure_dates=True`.")
    sauver = (lambda nom: str(Path(save_dir) / nom)) if save_dir else (lambda nom: None)

    # ---- 1. Univariée
    print("\n=== 1. Détection univariée (IQR) ===")
    flags, resume = detection_iqr(table, groupes, k_iqr, show=show, save_path=sauver("iqr_boxplots.png"))
    with pd.option_context("display.width", 200, "display.float_format", "{:,.4g}".format):
        print(resume.to_string())

    out = df.copy()
    for c in flags.columns:
        out[f"iqr_{c}"] = flags[c].to_numpy()
    out["n_iqr"] = flags.sum(axis=1).to_numpy()

    # ---- 2 et 3. Multivariée
    n_methodes = pd.Series(0, index=out.index)
    if table.shape[1] < 2 or len(table) < 30:
        print("\nDétection multivariée ignorée : il faut >= 2 variables et >= 30 lignes.")
    else:
        Z = matrice_multivariee(table, groupes)
        print("\n=== 2. Détection multivariée par regroupement ===")
        cl = detection_clustering(Z, eps=eps, min_samples=min_samples, k_iqr=k_iqr,
                                  contamination=contamination)
        print("\n=== 3. Angle-based Outlier Detection (ABOD) ===")
        ab = detection_abod(Z, k_abod, contamination)
        for nom in ("out_kmeans", "dist_centre", "out_dbscan"):
            out[nom] = cl[nom]
        out["out_abod"], out["abof"] = ab["out_abod"], ab["abof"]
        n_methodes = out[["out_kmeans", "out_dbscan", "out_abod"]].sum(axis=1)

    out["n_methodes"] = n_methodes
    out["anomalie_consensus"] = n_methodes >= min_methodes
    if "out_kmeans" in out and (show or save_dir):
        _graphique_multivarie(Z, out, sauver("multivarie_acp.png"), show)
    anomalies = out[out["anomalie_consensus"]].sort_values("n_methodes", ascending=False)

    print("\n=== 4. Synthèse ===")
    if "out_kmeans" in out:
        print("Lignes signalées :", {m: int(out[m].sum()) for m in ("out_kmeans", "out_dbscan", "out_abod")})
        print("Nombre de méthodes d'accord :", n_methodes.value_counts().sort_index().to_dict())
    print(f"Lignes en consensus (>= {min_methodes} méthodes) : {len(anomalies)}")
    print(f"Lignes avec au moins une valeur hors IQR : {int((out['n_iqr'] > 0).sum())}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out.to_csv(sauver("donnees_avec_anomalies.csv"), index=False)
        anomalies.to_csv(sauver("anomalies_consensus.csv"), index=False)
        resume.to_csv(sauver("resume_iqr.csv"))
        print(f"Résultats enregistrés dans : {save_dir}")

    return ResultatAnomalies(out, anomalies, resume, list(table.columns),
                             dict(group_by=group_by, contamination=contamination, k_abod=k_abod,
                                  min_methodes=min_methodes))


# ======================================================================
# Graphiques
# ======================================================================
def _finir(fig, save_path: Optional[str], show: bool):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _graphique_multivarie(Z: np.ndarray, out: pd.DataFrame, save_path: Optional[str], show: bool):
    """Projection ACP (2 axes) de l'espace des variables, une carte par méthode."""
    P = PCA(n_components=2).fit_transform(Z) if Z.shape[1] >= 2 else np.c_[Z[:, 0], np.zeros(len(Z))]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharex=True, sharey=True)
    for ax, col, titre in zip(axes, ["out_kmeans", "out_dbscan", "out_abod", "anomalie_consensus"],
                              ["K-means (distance au centre)", "DBSCAN (bruit)", "ABOD (FastABOD)",
                               "Consensus"]):
        flag = out[col].to_numpy(dtype=bool)
        ax.scatter(P[~flag, 0], P[~flag, 1], c="lightgray", s=12)
        ax.scatter(P[flag, 0], P[flag, 1], c="crimson", s=22)
        ax.set_title(f"{titre} : {int(flag.sum())}", fontsize=10)
        ax.set_xlabel("axe 1 (ACP)")
    axes[0].set_ylabel("axe 2 (ACP)")
    fig.suptitle("Détection multivariée : projection ACP des variables")
    fig.tight_layout()
    _finir(fig, save_path, show)


# ======================================================================
# Ligne de commande
# ======================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Détection d'anomalies (IQR, clustering, ABOD) sur un CSV.")
    ap.add_argument("fichier")
    ap.add_argument("--colonnes", nargs="+", help="colonnes à analyser (défaut : toutes les numériques)")
    ap.add_argument("--group", help="colonne de regroupement (ex. item_code)")
    ap.add_argument("--dates", action="store_true", help="inclure les dates (converties en jours)")
    ap.add_argument("--contamination", type=float, default=0.03, help="part signalée par l'ABOD (0.03)")
    ap.add_argument("--min-methodes", type=int, default=2, help="accord minimal entre méthodes (2)")
    ap.add_argument("--sortie", help="dossier de sauvegarde des CSV et des figures")
    ap.add_argument("--sans-graphiques", action="store_true", help="ne pas afficher les figures")
    a = ap.parse_args()

    if a.sans_graphiques:
        plt.switch_backend("Agg")
    donnees = pd.read_csv(a.fichier, sep=None, engine="python")
    r = detecter_anomalies(donnees, colonnes=a.colonnes, group_by=a.group, inclure_dates=a.dates,
                           contamination=a.contamination, min_methodes=a.min_methodes,
                           show=not a.sans_graphiques, save_dir=a.sortie)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nLignes en consensus :")
        print(r.anomalies.head(30).to_string(index=False) if len(r.anomalies) else "  aucune")
