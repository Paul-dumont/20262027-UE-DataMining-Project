from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.dates as mdates
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


def type_variable(serie: pd.Series, seuil_modalites: int = 20, seuil_ratio: float = 0.05) -> str:
    """
    'discrète' ou 'continue' pour une variable numérique.
    Discrète : valeurs toutes entières ET (<= seuil_modalites valeurs distinctes
    OU ratio valeurs distinctes / effectif <= seuil_ratio). Sinon continue.
    """
    s = serie.dropna()
    if s.empty:
        return "indéterminée"
    est_entier = np.allclose(s, np.round(s))
    n_unique = s.nunique()
    if est_entier and (n_unique <= seuil_modalites or n_unique / len(s) <= seuil_ratio):
        return "discrète"
    return "continue"


def _jours(serie_dates: pd.Series) -> pd.Series:
    """Dates -> nombre de jours depuis 1970 (permet de calculer moyenne, quartiles...)."""
    return (serie_dates - EPOCH) / pd.Timedelta(days=1)


def _vers_date(jours: float) -> pd.Timestamp:
    return (EPOCH + pd.to_timedelta(jours, unit="D")).round("min")


# ======================================================================
# Statistiques et graphiques d'une variable numérique ou date
# ======================================================================

def _stats_numeriques(s: pd.Series, type_var: str, est_date: bool = False) -> dict:
    """Statistiques toujours associées : (moyenne, écart-type) et (médiane, min, Q1, Q3, max)."""
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    conv = _vers_date if est_date else (lambda x: x)
    return {
        "type": type_var,
        "n": int(len(s)),
        "valeurs_distinctes": int(s.nunique()),
        "moyenne": conv(s.mean()),
        "ecart_type": s.std(),             # en jours pour les dates
        "min": conv(s.min()), "Q1": conv(q1), "mediane": conv(med),
        "Q3": conv(q3), "max": conv(s.max()),
        "asymetrie": s.skew(),
    }


def _formater_axe_date(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _histogramme(ax, s: pd.Series, type_var: str, bins, est_date: bool, taille_legende: int = 8):
    """Histogramme avec moyenne, bande ± écart-type et médiane."""
    m, sd, med = s.mean(), s.std(), s.median()
    fmt = (lambda x: _vers_date(x).strftime("%Y-%m-%d")) if est_date else (lambda x: f"{x:.2f}")

    if type_var == "discrète" and s.nunique() <= 50 and not est_date:
        valeurs = np.sort(s.unique())
        pas = np.min(np.diff(valeurs)) if len(valeurs) > 1 else 1
        bords = np.append(valeurs - pas / 2, valeurs[-1] + pas / 2)
        ax.hist(s, bins=bords, color="#4C72B0", edgecolor="white", rwidth=0.9)
    else:
        ax.hist(s, bins=bins, color="#4C72B0", edgecolor="white")

    ax.axvline(m, color="#C44E52", lw=2, label=f"moyenne = {fmt(m)}")
    if pd.notna(sd):
        unite = " j" if est_date else ""
        ax.axvspan(m - sd, m + sd, color="#C44E52", alpha=0.12,
                   label=f"± écart-type ({sd:.1f}{unite})")
    ax.axvline(med, color="#55A868", lw=2, ls="--", label=f"médiane = {fmt(med)}")
    ax.set_ylabel("Effectif")
    ax.legend(fontsize=taille_legende)
    if est_date:
        _formater_axe_date(ax)


def _tracer_numerique(s: pd.Series, nom: str, type_var: str, second_plot: str, bins,
                      est_date: bool, ax_hist, ax_box):
    """Histogramme (toujours) + boxplot ou violinplot, sur le même axe des valeurs."""
    _histogramme(ax_hist, s, type_var, bins, est_date)
    ax_hist.set_title(f"{nom}  ({'date' if est_date else type_var})")

    if second_plot == "violin" and s.nunique() > 1:
        parts = ax_box.violinplot(s, vert=False, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#8172B2")
            pc.set_alpha(0.6)
    else:
        ax_box.boxplot(s, vert=False, showmeans=True, widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor="#CCB974", alpha=0.7))
    ax_box.set_yticks([])
    ax_box.set_xlabel(nom)
    ax_box.set_title("violinplot" if second_plot == "violin" else "boxplot", fontsize=9)
    if est_date:
        _formater_axe_date(ax_box)

# ======================================================================
# Fonction 1 : analyse univariée détaillée
# ======================================================================
def analyser_variables(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    second_plot: str = "box",
    bins="auto",
    seuil_modalites: int = 20,
    seuil_ratio: float = 0.05,
    inclure_dates: bool = True,
    inclure_categorielles: bool = True,
    max_modalites_affichees: int = 15,
    convertir_textes: bool = True,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    if second_plot not in ("box", "violin"):
        raise ValueError("second_plot doit valoir 'box' ou 'violin'.")
    cols = [c for c in (columns if columns is not None else df.columns) if c in df.columns]
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    lignes: Dict[str, dict] = {}
    for col in cols:
        nature, serie = _nature_colonne(df[col], convertir_textes)
        s = serie.dropna()
        manquants = int(df[col].isna().sum())

        # ---------------- identifiant / vide ----------------
        if nature in ("vide", "identifiant"):
            print(f"\n=== {col}  [{nature}] ===")
            print("  Colonne vide." if nature == "vide" else
                  f"  {s.nunique()} valeurs toutes (quasi) différentes : non analysée.")
            lignes[col] = {"type": nature, "n": int(len(s)), "manquants": manquants,
                           "valeurs_distinctes": int(s.nunique())}
            continue

        # ---------------- catégorielle ----------------
        if nature == "catégorielle":
            if not inclure_categorielles:
                continue
            freq = s.astype(str).value_counts()
            pct = (freq / freq.sum() * 100).round(1)
            print(f"\n=== {col}  [catégorielle] ===")
            print(f"  Effectif : {len(s)}  (manquants : {manquants}, modalités : {len(freq)})")
            print(f"  Mode : {freq.index[0]} ({pct.iloc[0]} %)")
            print(pd.DataFrame({"effectif": freq, "%": pct}).head(max_modalites_affichees).to_string())
            lignes[col] = {"type": "catégorielle", "n": int(len(s)), "manquants": manquants,
                           "valeurs_distinctes": int(len(freq)),
                           "mode": freq.index[0], "freq_mode_%": float(pct.iloc[0])}

            top = freq.head(max_modalites_affichees)[::-1]
            fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(top) + 1.2)))
            barres = ax.barh(top.index.astype(str), top.values, color="#4C72B0")
            ax.bar_label(barres, labels=[f"{v} ({p} %)" for v, p in
                                         zip(top.values, pct.head(max_modalites_affichees)[::-1])],
                         fontsize=8, padding=3)
            ax.set_title(f"{col}  (catégorielle, {len(freq)} modalités)")
            ax.set_xlabel("Effectif")
            ax.set_xlim(0, top.max() * 1.25)
            if save_dir:
                fig.savefig(Path(save_dir) / f"{col}_distribution.png", dpi=120, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            continue

        # ---------------- numérique / date ----------------
        est_date = nature == "date"
        if est_date and not inclure_dates:
            continue
        valeurs = _jours(s) if est_date else s
        type_var = "date" if est_date else type_variable(s, seuil_modalites, seuil_ratio)
        st = _stats_numeriques(valeurs, type_var, est_date)
        st["manquants"] = manquants
        lignes[col] = st

        print(f"\n=== {col}  [{type_var}] ===")
        print(f"  Effectif : {st['n']}  (manquants : {manquants}, "
              f"valeurs distinctes : {st['valeurs_distinctes']})")
        if est_date:
            f = lambda d: d.strftime("%Y-%m-%d")
            print(f"  Moyenne ± écart-type : {f(st['moyenne'])} ± {st['ecart_type']:.1f} jours")
            print(f"  Médiane [min ; Q1 - Q3 ; max] : {f(st['mediane'])} "
                  f"[{f(st['min'])} ; {f(st['Q1'])} - {f(st['Q3'])} ; {f(st['max'])}]")
        else:
            print(f"  Moyenne ± écart-type : {st['moyenne']:.3f} ± {st['ecart_type']:.3f}")
            print(f"  Médiane [min ; Q1 - Q3 ; max] : {st['mediane']:.3f} "
                  f"[{st['min']:.3f} ; {st['Q1']:.3f} - {st['Q3']:.3f} ; {st['max']:.3f}]")
        print(f"  Asymétrie : {st['asymetrie']:.2f}")

        fig, (ax_h, ax_b) = plt.subplots(
            2, 1, figsize=(8, 5.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.2},
        )
        _tracer_numerique(valeurs, col, type_var, second_plot, bins, est_date, ax_h, ax_b)
        if save_dir:
            fig.savefig(Path(save_dir) / f"{col}_distribution.png", dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    if not lignes:
        print("Aucune variable à analyser.")
    return pd.DataFrame.from_dict(lignes, orient="index")


# ======================================================================
# Fonction 2 : vue d'ensemble de tous les histogrammes
# ======================================================================


def tracer_histogrammes(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    ncols: int = 3,
    bins="auto",
    inclure_categorielles: bool = True,
    max_modalites_affichees: int = 10,
    convertir_textes: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Trace UNE figure avec un graphique par variable :
      - histogramme pour les variables numériques et les dates
        (avec moyenne, bande ± écart-type et médiane),
      - diagramme en barres pour les variables catégorielles,
      - les identifiants (valeurs toutes différentes) sont ignorés.
    Retourne la figure matplotlib (ou None s'il n'y a rien à tracer).
    """
    cols = [c for c in (columns if columns is not None else df.columns) if c in df.columns]
    panneaux = []
    for col in cols:
        nature, serie = _nature_colonne(df[col], convertir_textes)
        if nature in ("numérique", "date") or (nature == "catégorielle" and inclure_categorielles):
            panneaux.append((col, nature, serie.dropna()))
    if not panneaux:
        print("Aucune variable à tracer.")
        return None

    nrows = math.ceil(len(panneaux) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (col, nature, s) in zip(axes.ravel(), panneaux):
        if nature == "catégorielle":
            freq = s.astype(str).value_counts().head(max_modalites_affichees)[::-1]
            barres = ax.barh(freq.index.astype(str), freq.values, color="#4C72B0")
            ax.bar_label(barres, fontsize=7, padding=2)
            ax.set_xlim(0, freq.max() * 1.18)
            ax.set_xlabel("Effectif")
            ax.set_title(f"{col}  (catégorielle)", fontsize=10)
        else:
            est_date = nature == "date"
            v = _jours(s) if est_date else s
            type_var = "date" if est_date else type_variable(v)
            _histogramme(ax, v, type_var, bins, est_date, taille_legende=7)
            ax.set_title(f"{col}  ({type_var})", fontsize=10)
    for ax in axes.ravel()[len(panneaux):]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python variable_analysis.py fichier.csv")
        sys.exit(1)
    data = pd.read_csv(sys.argv[1], sep=None, engine="python")
    tracer_histogrammes(data)
    analyser_variables(data)