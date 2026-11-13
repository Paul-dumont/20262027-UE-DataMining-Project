from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

# Formats de dates testés (ceux du notebook + variantes avec heure)
DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
)

# Valeurs textuelles à considérer comme "manquantes"
VALEURS_VIDES = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}


# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------
def _lire_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Lit un CSV en détectant automatiquement le séparateur et l'encodage."""
    kwargs.setdefault("sep", None)
    kwargs.setdefault("engine", "python")
    derniere_erreur = None
    for encodage in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encodage, **kwargs)
        except UnicodeDecodeError as e:
            derniere_erreur = e
    raise derniere_erreur


def parse_date(value):
    """Version du notebook : essaie plusieurs formats, puis un parsing par défaut."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if s == "":
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(datetime.strptime(s, fmt))
        except ValueError:
            pass
    return pd.to_datetime(s, errors="coerce")


def _parse_dates_serie(serie: pd.Series, fallback: bool = True) -> pd.Series:
    """Parsing vectorisé de dates à formats mixtes (équivalent rapide de parse_date)."""
    texte = serie.astype("string").str.strip()
    resultat = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    restant = texte.notna() & ~texte.str.lower().isin(VALEURS_VIDES)
    for fmt in DATE_FORMATS:
        if not restant.any():
            break
        essai = pd.to_datetime(texte[restant], format=fmt, errors="coerce")
        ok = essai.notna()
        resultat.loc[essai.index[ok]] = essai[ok]
        restant.loc[essai.index[ok]] = False
    if fallback and restant.any():
        resultat.loc[restant] = pd.to_datetime(texte[restant], errors="coerce")
    return resultat


def _est_colonne_date(serie: pd.Series, seuil: float = 0.8) -> bool:
    """Détecte une colonne de dates (>= seuil des valeurs parsées avec un format connu)."""
    valeurs = serie.dropna()
    valeurs = valeurs[~valeurs.astype(str).str.strip().str.lower().isin(VALEURS_VIDES)]
    if valeurs.empty:
        return False
    parsees = _parse_dates_serie(valeurs, fallback=False)
    return parsees.notna().mean() >= seuil


def _normaliser_cle(valeur: str) -> str:
    """Clé de regroupement : minuscules, espaces/tirets/underscores unifiés."""
    return re.sub(r"[\s\-_]+", "_", valeur.strip().lower())


def _colonne_cle_auto(df: pd.DataFrame) -> list:
    """Devine la/les colonne(s) clé(s) : nom type id/code/key et valeurs (quasi) uniques."""
    candidats = [
        c for c in df.columns
        if re.search(r"(^|_)(id|code|key|uuid|ref|number|no)($|_)", str(c).lower())
        or str(c).lower().endswith(("id", "code"))
    ]
    candidats = [c for c in candidats if df[c].nunique(dropna=True) / max(len(df), 1) > 0.5]
    # on privilégie la colonne la plus "unique"
    candidats.sort(key=lambda c: -df[c].nunique(dropna=True))
    return candidats[:1]


# ----------------------------------------------------------------------
# Fonction principale
# ----------------------------------------------------------------------
def clean_csv(
    source: Union[str, Path, pd.DataFrame],
    *,
    expected_columns: Optional[Iterable[str]] = None,
    key_columns: Optional[Iterable[str]] = None,
    required_columns: Optional[Iterable[str]] = None,
    date_columns: Optional[Iterable[str]] = None,
    categorical_columns: Optional[Iterable[str]] = None,
    category_mappings: Optional[Dict[str, Dict[str, str]]] = None,
    case_rules: Optional[Dict[str, str]] = None,
    max_categories: int = 50,
    convert_numeric: bool = True,
    outlier_method: Optional[str] = "iqr",
    outlier_factor: float = 3.0,
    numeric_fill: Optional[str] = None,
    categorical_fill: Optional[str] = "unknown",
    drop_rows_missing_required: bool = True,
    normalize_column_names: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    **read_csv_kwargs,
):
    log = print if verbose else (lambda *a, **k: None)
    rapport: dict = {}

    # --- Chargement -------------------------------------------------
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = _lire_csv(source, **read_csv_kwargs)
    rapport["forme_initiale"] = df.shape
    log(f"Avant nettoyage : {df.shape}")

    if normalize_column_names:
        df.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # ==================================================================
    # 1) REDONDANCES D'INFORMATIONS
    # ==================================================================
    # 1.a) Colonnes attendues
    if expected_columns is not None:
        manquantes = [c for c in expected_columns if c not in df.columns]
        rapport["colonnes_manquantes"] = manquantes
        log("Colonnes manquantes :" if manquantes else
            "Toutes les colonnes attendues sont présentes.", manquantes or "")

    # 1.b) Espaces dans les chaînes + chaînes vides / "nan" -> NaN
    colonnes_texte = df.select_dtypes(include=["object", "string"]).columns
    for col in colonnes_texte:
        s = df[col].astype("string").str.strip()
        s = s.mask(s.str.lower().isin(VALEURS_VIDES))
        df[col] = s.astype(object).where(s.notna(), np.nan)

    # 1.c) Doublons (lignes entièrement identiques, puis doublons sur la clé)
    cles = list(key_columns) if key_columns is not None else _colonne_cle_auto(df)
    cles = [c for c in cles if c in df.columns]
    rapport["colonnes_cles"] = cles

    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    n_cle = 0
    if cles:
        n_cle = int(df.duplicated(subset=cles).sum())
        log(f"Nombre de doublons sur {cles} : {n_cle}")
        df = df.drop_duplicates(subset=cles).copy()
    rapport["doublons_exacts_supprimes"] = n_exact
    rapport["doublons_cle_supprimes"] = n_cle
    log(f"Après suppression des doublons : {df.shape}")

    # ==================================================================
    # 2) FORMATS INCONSISTANTS (dates, nombres)
    # ==================================================================
    cols_dates = list(date_columns) if date_columns is not None else [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in cles and _est_colonne_date(df[c])
    ]
    cols_dates = [c for c in cols_dates if c in df.columns]
    for col in cols_dates:
        df[col] = _parse_dates_serie(df[col])
    rapport["colonnes_dates"] = cols_dates
    if cols_dates:
        log("Dates normalisées :", {c: str(df[c].dtype) for c in cols_dates})

    cols_numeriques_converties = []
    if convert_numeric:
        for col in df.select_dtypes(include=["object"]).columns:
            if col in cles or col in cols_dates:
                continue
            valeurs = df[col].dropna().astype(str)
            if valeurs.empty or valeurs.str.match(r"^0\d+$").any():  # codes type "007"
                continue
            nettoyees = valeurs.str.replace(r"[\s\u00a0]", "", regex=True)
            # "1 234,56" -> "1234.56"
            nettoyees = nettoyees.str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True)
            nombres = pd.to_numeric(nettoyees, errors="coerce")
            if nombres.notna().mean() >= 0.9:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[\s\u00a0]", "", regex=True)
                    .str.replace(r"^(-?\d+),(\d+)$", r"\1.\2", regex=True),
                    errors="coerce",
                )
                cols_numeriques_converties.append(col)
    rapport["colonnes_numeriques_converties"] = cols_numeriques_converties

    # ==================================================================
    # 3) VARIABLES CATÉGORIELLES
    # ==================================================================
    if categorical_columns is not None:
        cols_cat = [c for c in categorical_columns if c in df.columns]
    else:
        cols_cat = [
            c for c in df.select_dtypes(include=["object"]).columns
            if c not in cles and c not in cols_dates
            and df[c].nunique(dropna=True) <= max_categories
            and (df[c].nunique(dropna=True) < 0.5 * max(len(df), 1)
                 or df[c].nunique(dropna=True) <= 10)
        ]
    case_rules = case_rules or {}
    category_mappings = category_mappings or {}
    modifs_categories = {}

    for col in cols_cat:
        avant = df[col].nunique(dropna=True)
        serie = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

        if col in case_rules:
            regle = case_rules[col]
            serie = {"lower": serie.str.lower(), "upper": serie.str.upper(),
                     "title": serie.str.title()}[regle]
        else:
            # Regroupe les variantes (casse, "-" vs "_", espaces) et retient la
            # forme la plus fréquente comme libellé canonique (cf. mapping_regions).
            valeurs = serie.dropna()
            if not valeurs.empty:
                cles_norm = valeurs.map(_normaliser_cle)
                tableau = pd.DataFrame({"orig": valeurs, "norm": cles_norm})
                canonique = (
                    tableau.groupby("norm")["orig"]
                    .agg(lambda x: x.value_counts().index[0])
                )
                serie = serie.where(serie.isna(), serie.map(lambda v: canonique[_normaliser_cle(v)]
                                                            if pd.notna(v) else v))
        df[col] = serie.astype(object).where(serie.notna(), np.nan)

        if col in category_mappings:
            df[col] = df[col].replace(category_mappings[col])
        modifs_categories[col] = (avant, df[col].nunique(dropna=True))

    rapport["colonnes_categorielles"] = cols_cat
    rapport["categories_avant_apres"] = modifs_categories
    for col, (a, b) in modifs_categories.items():
        if a != b:
            log(f"{col} : {a} -> {b} modalités distinctes")

    # ==================================================================
    # 4) DONNÉES MANQUANTES ET ABERRANTES
    # ==================================================================
    # 4.a) Suppression des lignes dont la clé principale manque
    if cles:
        avant = len(df)
        df = df.dropna(subset=cles).copy()
        rapport["lignes_supprimees_cle_manquante"] = avant - len(df)

    # 4.b) Valeurs aberrantes -> NaN
    cols_num = df.select_dtypes(include=[np.number]).columns.difference(cles)
    outliers = {}
    if outlier_method in ("iqr", "zscore"):
        for col in cols_num:
            s = df[col]
            if s.nunique(dropna=True) < 3:
                continue
            if outlier_method == "iqr":
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                masque = (s < q1 - outlier_factor * iqr) | (s > q3 + outlier_factor * iqr)
            else:
                ecart = s.std()
                masque = ((s - s.mean()).abs() / ecart > outlier_factor) if ecart else s.isna() & False
            n = int(masque.sum())
            if n:
                df.loc[masque, col] = np.nan
                outliers[col] = n
    # Dates aberrantes (avant 1900 ou dans le futur)
    for col in cols_dates:
        masque = (df[col] < "1900-01-01") | (df[col] > pd.Timestamp.now())
        n = int(masque.sum())
        if n:
            df.loc[masque, col] = pd.NaT
            outliers[col] = n
    rapport["outliers_traites_comme_manquants"] = outliers
    if outliers:
        log("Valeurs aberrantes mises à NaN :", outliers)

    # 4.c) Rapport des valeurs manquantes avant traitement
    rapport["manquants_avant_traitement"] = df.isna().sum()[lambda s: s > 0].to_dict()

    # 4.d) Option A : suppression des lignes si colonnes requises manquantes
    if required_columns is not None:
        req = [c for c in required_columns if c in df.columns]
        if drop_rows_missing_required and req:
            avant = len(df)
            df = df.dropna(subset=req).copy()
            rapport["lignes_supprimees_colonnes_requises"] = avant - len(df)

    # 4.e) Option B : remplissage ("unknown" pour les catégories)
    if categorical_fill is not None:
        for col in cols_cat:
            df[col] = df[col].fillna(categorical_fill)
        # Autres colonnes texte (non catégorielles) : même traitement
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in cles:
                df[col] = df[col].fillna(categorical_fill)

    if numeric_fill:
        for col in df.select_dtypes(include=[np.number]).columns.difference(cles):
            valeur = {"median": df[col].median(), "mean": df[col].mean(), "zero": 0}[numeric_fill]
            df[col] = df[col].fillna(valeur)

    # --- Rapport final ---------------------------------------------
    df = df.reset_index(drop=True)
    rapport["forme_finale"] = df.shape
    rapport["manquants_apres_traitement"] = df.isna().sum()[lambda s: s > 0].to_dict()
    log(f"\nForme finale : {df.shape}")
    log("Valeurs manquantes après traitement :")
    log(df.isna().sum()[lambda s: s > 0] if df.isna().any().any() else "Aucune")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log(f"Fichier sauvegardé : {output_path}")

    return df, rapport


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python clean_csv.py fichier.csv [sortie.csv]")
        sys.exit(1)
    sortie = sys.argv[2] if len(sys.argv) > 2 else None
    clean_csv(sys.argv[1], output_path=sortie)