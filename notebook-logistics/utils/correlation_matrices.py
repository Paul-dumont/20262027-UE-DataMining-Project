from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
)
VALEURS_VIDES = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}
EPOCH = pd.Timestamp("1970-01-01")

# ======================================================================
# Détection de la nature des colonnes
# ======================================================================
def _convertir_nombres(serie: pd.Series, seuil: float = 0.9) -> Optional[pd.Series]:
    """Convertit une colonne texte en nombres si >= seuil des valeurs sont numériques."""
    t = serie.astype("string").str.strip()
    t = t.mask(t.str.lower().isin(VALEURS_VIDES))
    valides = t.dropna()
    if valides.empty or valides.str.match(r"^0\d+$").any():  # codes du type "007"
        return None
    t = t.str.replace(r"[\s\u00a0]", "", regex=True)
    t = t.str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True)
    nombres = pd.to_numeric(t, errors="coerce").astype(float)
    return nombres if nombres.notna().sum() / len(valides) >= seuil else None


def _convertir_dates(serie: pd.Series, seuil: float = 0.8) -> Optional[pd.Series]:
    """Convertit une colonne texte en dates (formats mixtes) si >= seuil sont reconnues."""
    t = serie.astype("string").str.strip().reset_index(drop=True)
    valides = t.notna() & ~t.str.lower().isin(VALEURS_VIDES)
    if not valides.any():
        return None
    res = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    restant = valides.copy()
    for fmt in DATE_FORMATS:
        if not restant.any():
            break
        essai = pd.to_datetime(t[restant], format=fmt, errors="coerce")
        ok = essai.notna()
        res.loc[essai.index[ok]] = essai[ok]
        restant.loc[essai.index[ok]] = False
    if res.notna().sum() / valides.sum() >= seuil:
        res.index = serie.index
        return res
    return None


def _nature_colonne(serie: pd.Series, convertir: bool = True):
    """
    Renvoie (nature, serie_convertie) avec nature parmi :
    'numérique', 'date', 'catégorielle', 'identifiant', 'vide'.
    """
    if serie.dropna().empty:
        return "vide", serie
    if pd.api.types.is_bool_dtype(serie):
        return "catégorielle", serie
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "date", serie
    if pd.api.types.is_numeric_dtype(serie):
        return "numérique", serie.astype(float)
    if convertir:
        nombres = _convertir_nombres(serie)
        if nombres is not None:
            return "numérique", nombres
        dates = _convertir_dates(serie)
        if dates is not None:
            return "date", dates
    s = serie.astype("object")
    n_valides = int(s.notna().sum())
    if n_valides > 5 and s.nunique() / n_valides > 0.9:
        return "identifiant", s
    return "catégorielle", s



def _jours(serie_dates: pd.Series) -> pd.Series:
    """Dates -> nombre de jours depuis 1970 (permet de calculer moyenne, quartiles...)."""
    return (serie_dates - EPOCH) / pd.Timedelta(days=1)



# ======================================================================
# Utilitaires de heatmap
# ======================================================================
def _heatmap(matrice: pd.DataFrame, titre: str, label: str, vmin: float, vmax: float,
             cmap: str, annot: bool, masquer_triangle: bool, figsize, show: bool,
             save_path: Optional[str]):
    n = matrice.shape[0]
    taille = figsize or (max(6, 0.8 * n + 2), max(5, 0.7 * n + 1.5))
    fig, ax = plt.subplots(figsize=taille)
    valeurs = matrice.to_numpy(dtype=float).copy()
    if masquer_triangle:
        valeurs[np.triu_indices(n, k=1)] = np.nan
    im = ax.imshow(valeurs, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrice.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrice.index)
    if annot:
        for i in range(n):
            for j in range(n):
                v = valeurs[i, j]
                if not np.isnan(v):
                    if vmin < 0:   # palette divergente (corrélation) : fond foncé aux extrêmes
                        fonce = abs(v - (vmin + vmax) / 2) > 0.3 * (vmax - vmin)
                    else:          # palette séquentielle (Cramér) : fond foncé pour les fortes valeurs
                        fonce = v > vmin + 0.6 * (vmax - vmin)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if fonce else "black")
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(titre)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _afficher_paires(matrice: pd.DataFrame, seuil: float, nom: str):
    paires = (
        matrice.where(np.tril(np.ones(matrice.shape, dtype=bool), k=-1))
        .stack().rename("valeur").reset_index()
        .rename(columns={"level_0": "variable_1", "level_1": "variable_2"})
    )
    fortes = paires[paires["valeur"].abs() >= seuil].sort_values("valeur", key=lambda x: -x.abs())
    print(f"\nPaires avec |{nom}| >= {seuil} :")
    print(fortes.to_string(index=False) if not fortes.empty else "  aucune")


# ======================================================================
# Fonction 1 : matrice de corrélation (Spearman par défaut)
# ======================================================================
def matrice_correlation(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    method: str = "spearman",
    ordres: Optional[Dict[str, List[str]]] = None,
    inclure_dates: bool = True,
    convertir_textes: bool = True,
    fallback_categorielles: bool = True,
    seuil_paires: float = 0.7,
    annot: bool = True,
    masquer_triangle: bool = True,
    figsize: Optional[tuple] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Matrice de corrélation (Spearman par défaut) entre :
      - les variables numériques (y compris colonnes texte convertibles en nombres),
      - les dates (converties en nombre de jours),
      - les variables ORDINALES déclarées dans `ordres`, par exemple
            ordres={"risk_class": ["low", "standard", "review", "elevated"],
                    "partner_status": ["inactive", "active"]}
        (l'ordre de la liste = ordre croissant ; casse ignorée).

    Si moins de 2 variables exploitables et fallback_categorielles=True, la fonction
    bascule automatiquement sur matrice_cramer() (association entre catégories).
    """
    cols = [c for c in (columns if columns is not None else df.columns) if c in df.columns]
    ordres = {c: [str(v).strip().lower() for v in o] for c, o in (ordres or {}).items()}

    data, utilisees, exclues = {}, [], []
    for c in cols:
        if c in ordres:
            ref = {v: i for i, v in enumerate(ordres[c])}
            texte = df[c].astype("string").str.strip().str.lower()
            inconnues = sorted(set(texte.dropna()) - set(ref))
            if inconnues:
                print(f"[{c}] valeurs absentes de `ordres` (mises à NaN) : {inconnues}")
            data[c] = pd.to_numeric(texte.map(ref), errors="coerce").astype(float)
            utilisees.append(f"{c} (ordinale)")
            continue
        nature, serie = _nature_colonne(df[c], convertir_textes)
        if nature == "numérique":
            data[c] = serie
            utilisees.append(f"{c} (numérique)")
        elif nature == "date" and inclure_dates:
            data[c] = _jours(serie)
            utilisees.append(f"{c} (date -> jours)")
        else:
            exclues.append(f"{c} ({nature})")

    table = pd.DataFrame(data)
    constantes = [c for c in table.columns if table[c].nunique(dropna=True) < 2]
    if constantes:
        print(f"Colonnes constantes ou vides exclues : {constantes}")
        table = table.drop(columns=constantes)

    if table.shape[1] < 2:
        message = (
            f"Moins de 2 variables numériques/ordinales exploitables "
            f"(trouvées : {list(table.columns)}). Exclues : {exclues}."
        )
        if fallback_categorielles:
            print(message)
            print("-> Bascule sur la matrice de Cramér's V (association entre catégories).\n"
                  "   Pour utiliser Spearman, déclarez les variables ordonnées via `ordres=`.")
            return matrice_cramer(df, columns=columns, annot=annot,
                                  masquer_triangle=masquer_triangle, figsize=figsize,
                                  show=show, save_path=save_path)
        raise ValueError(message + " Utilisez `ordres=` ou matrice_cramer().")

    print(f"Variables utilisées ({table.shape[1]}) : {utilisees}")
    if exclues:
        print(f"Variables exclues : {exclues}")

    corr = table.corr(method=method)
    _heatmap(corr, f"Matrice de corrélation ({method.capitalize()})",
             f"Coefficient de {method.capitalize()}", -1, 1, "coolwarm",
             annot, masquer_triangle, figsize, show, save_path)
    _afficher_paires(corr, seuil_paires, method)
    return corr


# ======================================================================
# Fonction 2 : association entre variables catégorielles (Cramér's V)
# ======================================================================
def _cramer_v(x: pd.Series, y: pd.Series, corriger: bool = True) -> float:
    """V de Cramér (0 = indépendance, 1 = association parfaite), avec correction de biais."""
    tab = pd.crosstab(x, y)
    n = tab.values.sum()
    r, k = tab.shape
    if r < 2 or k < 2 or n < 2:
        return np.nan
    obs = tab.values.astype(float)
    attendu = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / n
    phi2 = ((obs - attendu) ** 2 / attendu).sum() / n
    if corriger:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_c = r - (r - 1) ** 2 / (n - 1)
        k_c = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_c - 1, r_c - 1)
    else:
        denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def matrice_cramer(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    max_modalites: int = 30,
    corriger_biais: bool = True,
    seuil_paires: float = 0.5,
    annot: bool = True,
    masquer_triangle: bool = True,
    figsize: Optional[tuple] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Matrice de V de Cramér entre variables catégorielles (valeurs de 0 à 1).
    Sont exclus : identifiants, dates, colonnes constantes et colonnes
    ayant plus de `max_modalites` modalités.
    """
    cols = [c for c in (columns if columns is not None else df.columns) if c in df.columns]
    cats, exclues = {}, []
    for c in cols:
        nature, serie = _nature_colonne(df[c], convertir=True)
        if nature == "numérique" and serie.nunique() <= 10:
            nature = "catégorielle"  # numérique à très peu de valeurs : traité comme catégorie
        if nature != "catégorielle":
            exclues.append(f"{c} ({nature})")
        elif serie.nunique() < 2 or serie.nunique() > max_modalites:
            exclues.append(f"{c} ({serie.nunique()} modalités)")
        else:
            cats[c] = serie.astype(object)
    if len(cats) < 2:
        raise ValueError(f"Moins de 2 variables catégorielles exploitables. Exclues : {exclues}")

    print(f"Variables catégorielles utilisées ({len(cats)}) : {list(cats)}")
    if exclues:
        print(f"Variables exclues : {exclues}")

    noms = list(cats)
    v = pd.DataFrame(np.eye(len(noms)), index=noms, columns=noms)
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            paire = pd.concat([cats[a], cats[b]], axis=1).dropna()
            v.loc[a, b] = v.loc[b, a] = _cramer_v(paire.iloc[:, 0], paire.iloc[:, 1], corriger_biais)

    _heatmap(v, "Association entre variables catégorielles (V de Cramér)",
             "V de Cramér", 0, 1, "Blues", annot, masquer_triangle, figsize, show, save_path)
    _afficher_paires(v, seuil_paires, "V de Cramér")
    return v

# ======================================================================
# Fonction 3 : tracer toutes les matrices
# ======================================================================


def tracer_matrices(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    ordres: Optional[Dict[str, List[str]]] = None,
    method: str = "spearman",
    show: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Trace les matrices possibles pour ce DataFrame :
      - corrélation (Spearman par défaut) si >= 2 variables numériques / dates / ordinales,
      - V de Cramér si >= 2 variables catégorielles.
    Retourne {"spearman": DataFrame, "cramer": DataFrame} (seulement celles calculées).
    """
    sauver = (lambda nom: str(Path(save_dir) / nom)) if save_dir else (lambda nom: None)
    resultats = {}
    try:
        resultats[method] = matrice_correlation(
            df, columns, method=method, ordres=ordres, fallback_categorielles=False,
            show=show, save_path=sauver(f"matrice_{method}.png"))
    except ValueError as e:
        print(f"Matrice de {method} non tracée : {e}")
    try:
        resultats["cramer"] = matrice_cramer(
            df, columns, show=show, save_path=sauver("matrice_cramer.png"))
    except ValueError as e:
        print(f"Matrice de Cramér non tracée : {e}")
    return resultats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python correlation_matrices.py fichier.csv")
        sys.exit(1)
    tracer_matrices(pd.read_csv(sys.argv[1], sep=None, engine="python"))