

from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

VALEURS_VIDES = {"", "nan", "none", "null", "na", "n/a", "nat", "<na>"}
_RE_NOMBRE = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*([A-Za-z_]+)?\s*$")


# ======================================================================
# Lecture et détection de la nature des colonnes
# ======================================================================
def lire_csv(chemin: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Lit un CSV en texte brut (séparateur détecté automatiquement). Rien n'est converti."""
    kwargs.setdefault("sep", None)
    kwargs.setdefault("engine", "python")
    kwargs.setdefault("dtype", str)
    return pd.read_csv(chemin, **kwargs)


def _texte_propre(serie: pd.Series) -> pd.Series:
    """Texte sans espaces superflus, valeurs vides -> NaN."""
    t = serie.astype("string").str.strip()
    return t.mask(t.str.lower().isin(VALEURS_VIDES))


def _convertir_nombres(serie: pd.Series, seuil: float = 0.9) -> Optional[pd.Series]:
    """
    Convertit une colonne texte en nombres si >= seuil des valeurs sont numériques.
    Accepte : virgule décimale (« 12,5 »), espaces de milliers, nombre suivi d'une unité
    (« 88 case »). Refuse les codes à zéros initiaux (« 007 »).
    """
    t = _texte_propre(serie)
    valides = t.dropna()
    if valides.empty or valides.str.match(r"^0\d+$").any():
        return None
    valides = valides.str.replace(r"(?<=\d)[\s\u00a0](?=\d{3}\b)", "", regex=True)
    ext = valides.str.extract(_RE_NOMBRE)
    if ext[0].notna().sum() / len(valides) < seuil:
        return None
    nombres = pd.to_numeric(ext[0].str.replace(",", ".", regex=False), errors="coerce")
    res = pd.Series(np.nan, index=serie.index, dtype=float)
    res.loc[nombres.index] = nombres.astype(float)
    return res


def _convertir_dates(serie: pd.Series, seuil: float = 0.8) -> Optional[pd.Series]:
    """Convertit une colonne texte en dates (formats mixtes) si >= seuil sont reconnues."""
    t = _texte_propre(serie)
    valides = t.dropna()
    if valides.empty:
        return None
    # garde-fou : une date contient des chiffres ET un séparateur
    if valides.str.contains(r"\d").mean() < 0.9 or valides.str.contains(r"[-/:.T ]").mean() < 0.9:
        return None
    try:
        dt = pd.to_datetime(t, format="mixed", errors="coerce", dayfirst=False)
    except (TypeError, ValueError):
        dt = pd.to_datetime(t, errors="coerce")
    if dt.notna().sum() / len(valides) >= seuil:
        return dt
    return None


def _nature_colonne(serie: pd.Series, convertir: bool = True):
    """Renvoie (nature, serie_convertie) ; nature ∈ numérique / date / catégorielle / identifiant / vide."""
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
    s = _texte_propre(serie).astype(object)
    n_valides = int(s.notna().sum())
    if n_valides > 5 and s.nunique() / n_valides > 0.9:
        return "identifiant", s
    return "catégorielle", s


def _normaliser_texte(s: pd.Series) -> pd.Series:
    """Minuscules, espaces multiples réduits : « Delivered », « DELIVERED » -> « delivered »."""
    s = s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.isin(VALEURS_VIDES))


# ======================================================================
# Jointure avec une autre table (variables dérivées)
# ======================================================================
def enrichir(
    df: pd.DataFrame,
    autre: Union[pd.DataFrame, str, Path],
    cle: str,
    colonnes: Sequence[str],
    cle_autre: Optional[str] = None,
    prefixe: str = "",
) -> pd.DataFrame:
    """
    Ajoute à `df` des colonnes d'une autre table (jointure à gauche sur une clé).
    Exemple : enrichir(shipments, facilities, cle="destination_facility_code",
                       cle_autre="facility_code", colonnes=["region"], prefixe="dest_")
    """
    if not isinstance(autre, pd.DataFrame):
        autre = lire_csv(autre)
    cle_autre = cle_autre or cle
    ref = autre.drop_duplicates(subset=[cle_autre]).set_index(cle_autre)
    sortie = df.copy()
    for c in colonnes:
        sortie[f"{prefixe}{c}"] = sortie[cle].map(ref[c])
    return sortie


# ======================================================================
# Étape 1 : DataFrame -> transactions
# ======================================================================
def construire_transactions(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    exclure: Iterable[str] = (),
    max_modalites: int = 30,
    n_classes: int = 3,
    dates: Sequence[str] = ("weekend",),
    max_valeurs_discretes: int = 10,
    verbose: bool = True,
) -> Tuple[List[FrozenSet[str]], pd.DataFrame]:

    exclure = set(exclure)
    cols = [c for c in (columns if columns is not None else df.columns) if c in df.columns and c not in exclure]
    series_items: Dict[str, pd.Series] = {}
    rapport = []

    def _ajouter(nom, items, nature, traitement):
        series_items[nom] = items.astype(object).where(items.notna(), None)
        rapport.append({"colonne": nom, "nature": nature, "traitement": traitement,
                        "n_items": int(items.nunique())})

    def _exclure(c, nature, raison):
        rapport.append({"colonne": c, "nature": nature, "traitement": f"exclue ({raison})", "n_items": 0})

    for c in cols:
        nature, s = _nature_colonne(df[c])
        if nature in ("vide", "identifiant"):
            _exclure(c, nature, nature)
        elif nature == "numérique":
            s = s.astype(float)
            n_u = s.nunique()
            if n_u < 2:
                _exclure(c, nature, "constante")
            elif n_u <= max_valeurs_discretes:
                _ajouter(c, s.map(lambda v: f"{v:g}" if pd.notna(v) else np.nan), nature, "valeurs discrètes")
            else:
                codes, bornes = pd.qcut(s, n_classes, labels=False, duplicates="drop", retbins=True)
                if len(bornes) < 3:
                    _exclure(c, nature, "constante après découpage")
                    continue
                etiq = {i: f"[{bornes[i]:.4g} .. {bornes[i + 1]:.4g}]" for i in range(len(bornes) - 1)}
                _ajouter(c, codes.map(etiq), nature, f"{len(etiq)} classes par quantiles")
        elif nature == "date":
            dt = s
            derivees = {
                "weekend": lambda x: x.dt.dayofweek.map(lambda v: np.nan if pd.isna(v) else ("oui" if v >= 5 else "non")),
                "jour_semaine": lambda x: x.dt.day_name(),
                "mois": lambda x: x.dt.month.map(lambda v: np.nan if pd.isna(v) else f"{int(v):02d}"),
                "trimestre": lambda x: x.dt.quarter.map(lambda v: np.nan if pd.isna(v) else f"T{int(v)}"),
                "annee": lambda x: x.dt.year.map(lambda v: np.nan if pd.isna(v) else str(int(v))),
            }
            for d in dates:
                if d not in derivees:
                    raise ValueError(f"dates : '{d}' inconnu. Choix : {list(derivees)}")
                it = derivees[d](dt)
                if it.nunique() >= 2:
                    _ajouter(f"{c}_{d}", it, "date", f"dérivée '{d}'")
        else:  # catégorielle
            t = _normaliser_texte(s)
            n_u = t.nunique()
            if n_u < 2:
                _exclure(c, nature, "constante")
            elif n_u > max_modalites:
                _exclure(c, nature, f"{n_u} modalités > {max_modalites}")
            else:
                _ajouter(c, t, nature, "modalités")

    rapport_df = pd.DataFrame(rapport)
    if not series_items:
        raise ValueError("Aucune colonne exploitable pour construire des transactions.")
    noms = list(series_items)
    tableaux = [series_items[n].tolist() for n in noms]
    transactions = [frozenset(f"{n}={v}" for n, v in zip(noms, ligne) if v is not None)
                    for ligne in zip(*tableaux)]
    if verbose:
        n_items = len({i for t in transactions for i in t})
        print(f"Transactions : {len(transactions)} | colonnes utilisées : {len(noms)} | items distincts : {n_items}")
        exclues = rapport_df[rapport_df.traitement.str.startswith("exclue")]
        if not exclues.empty:
            print("Colonnes exclues :", [f"{r.colonne} ({r.traitement[8:-1]})" for r in exclues.itertuples()])
    return transactions, rapport_df


# ======================================================================
# Étape 2 : Apriori (supports par masques de bits)
# ======================================================================
def _bit_count(x: int) -> int:
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


def apriori(
    transactions: Sequence[FrozenSet[str]],
    min_support: float = 0.05,
    min_abs: Optional[int] = None,
    max_len: Optional[int] = None,
    max_itemsets: int = 500_000,
) -> Dict[FrozenSet[str], int]:
    """
    Itemsets fréquents -> {itemset: support_absolu}.
    Anti-monotonie : un sur-ensemble d'un itemset non fréquent ne peut pas être fréquent,
    donc les candidats de taille k sont formés à partir des fréquents de taille k-1 et élagués
    si l'un de leurs sous-ensembles n'est pas fréquent.
    `min_abs` (support absolu minimum) remplace `min_support` s'il est donné.
    """
    n = len(transactions)
    if n == 0:
        raise ValueError("Base de transactions vide.")
    seuil = int(min_abs) if min_abs is not None else max(1, math.ceil(min_support * n - 1e-9))

    positions: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(transactions):
        for it in t:
            positions[it].append(i)
    labels = sorted(positions)
    indice = {it: i for i, it in enumerate(labels)}

    def _masque(pos):
        a = np.zeros(n, dtype=bool)
        a[pos] = True
        return int.from_bytes(np.packbits(a).tobytes(), "big")

    niveau: Dict[tuple, Tuple[int, int]] = {}
    for it in labels:
        if len(positions[it]) >= seuil:
            niveau[(indice[it],)] = (_masque(positions[it]), len(positions[it]))
    resultat = {k: v[1] for k, v in niveau.items()}

    k = 1
    while niveau and (max_len is None or k < max_len):
        k += 1
        groupes: Dict[tuple, List[int]] = defaultdict(list)
        for iset in niveau:
            groupes[iset[:-1]].append(iset[-1])
        nouveau: Dict[tuple, Tuple[int, int]] = {}
        for prefixe, fins in groupes.items():
            fins.sort()
            for a, b in combinations(fins, 2):
                cand = prefixe + (a, b)
                if k > 2 and any(cand[:j] + cand[j + 1:] not in niveau for j in range(k - 2)):
                    continue
                m = niveau[prefixe + (a,)][0] & niveau[prefixe + (b,)][0]
                c = _bit_count(m)
                if c >= seuil:
                    nouveau[cand] = (m, c)
        niveau = nouveau
        resultat.update({kk: v[1] for kk, v in niveau.items()})
        if len(resultat) > max_itemsets:
            raise ValueError(f"Plus de {max_itemsets} itemsets : augmentez min_support ou fixez max_len.")
    return {frozenset(labels[i] for i in iset): c for iset, c in resultat.items()}


def itemsets_frequents(
    transactions: Sequence[FrozenSet[str]],
    min_support: float = 0.05,
    min_abs: Optional[int] = None,
    max_len: Optional[int] = None,
    max_itemsets: int = 500_000,
) -> pd.DataFrame:
    """
    Itemsets fréquents sous forme de DataFrame :
    itemset (tuple trié), taille, support_abs, support_rel, lift (NaN pour la taille 1).
    lift = support(X) / produit des supports des items de X (1 = indépendance ; >> 1 = les items
    apparaissent ensemble bien plus que ne le prévoirait le hasard).
    """
    n = len(transactions)
    F = apriori(transactions, min_support, min_abs, max_len, max_itemsets)
    simples = {next(iter(x)): c for x, c in F.items() if len(x) == 1}
    lignes = []
    for x, c in F.items():
        lift = np.nan
        if len(x) >= 2:
            lift = (c / n) / math.prod(simples[i] / n for i in x)
        lignes.append((tuple(sorted(x)), len(x), c, c / n, lift))
    df = pd.DataFrame(lignes, columns=["itemset", "taille", "support_abs", "support_rel", "lift"])
    df = df.sort_values(["taille", "support_abs"], ascending=[True, False]).reset_index(drop=True)
    df.attrs["N"] = n
    return df


# ======================================================================
# Étape 3 : filtres, patterns surprenants, règles
# ======================================================================
def filtrer_fermes_maximaux(F: pd.DataFrame, mode: str = "ferme") -> pd.DataFrame:
    """
    Réduit la redondance.
      mode="ferme"   : garde les itemsets sans sur-ensemble de MÊME support (aucune perte d'information) ;
      mode="maximal" : garde ceux sans aucun sur-ensemble fréquent (représentation la plus courte).
    (Si apriori a été limité par max_len, "maximal" est approximatif.)
    """
    d = {frozenset(t): c for t, c in zip(F.itemset, F.support_abs)}
    a_sup, a_sup_meme = set(), set()
    for x, c in d.items():
        if len(x) >= 2:
            for it in x:
                sous = x - {it}
                if sous in d:
                    a_sup.add(sous)
                    if d[sous] == c:
                        a_sup_meme.add(sous)
    exclus = a_sup_meme if mode == "ferme" else a_sup
    garde = [frozenset(t) not in exclus for t in F.itemset]
    sortie = F[garde].reset_index(drop=True)
    sortie.attrs.update(F.attrs)
    return sortie


def itemsets_surprenants(
    F: pd.DataFrame,
    min_lift: float = 3.0,
    min_abs: int = 10,
    min_taille: int = 2,
    contient: Union[None, str, Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Itemsets à fort lift (triés par lift décroissant). `min_abs` évite d'interpréter du bruit
    (un lift énorme sur 3 lignes ne prouve rien). `contient` : ne garde que les itemsets dont
    un item contient ce texte (ou l'un de ces textes).
    """
    d = F[(F.taille >= min_taille) & (F.support_abs >= min_abs) & (F.lift >= min_lift)]
    if contient is not None:
        mots = [contient] if isinstance(contient, str) else list(contient)
        d = d[d.itemset.apply(lambda t: any(m in i for i in t for m in mots))]
    return d.sort_values(["lift", "support_abs"], ascending=False).reset_index(drop=True)


def regles_association(
    F: pd.DataFrame,
    min_confidence: float = 0.7,
    min_lift: float = 1.2,
    min_abs: Optional[int] = None,
    max_taille: int = 5,
    consequent_unique: bool = True,
) -> pd.DataFrame:
    """
    Règles A -> B à partir des itemsets fréquents (F issu de itemsets_frequents, SANS filtre).
        confiance = supp(A ∪ B) / supp(A)      (estimation de P(B | A))
        lift      = confiance / supp(B)
    consequent_unique=True : le conséquent est un seul item (évite l'explosion combinatoire).
    """
    n = F.attrs.get("N") or int(round(F.support_abs.iloc[0] / F.support_rel.iloc[0]))
    d = {frozenset(t): c for t, c in zip(F.itemset, F.support_abs)}
    lignes = []
    for x, c in d.items():
        if len(x) < 2 or len(x) > max_taille or (min_abs is not None and c < min_abs):
            continue
        if consequent_unique:
            partitions = [(x - {i}, frozenset([i])) for i in x]
        else:
            partitions = [(frozenset(a), x - frozenset(a)) for r in range(1, len(x))
                          for a in combinations(sorted(x), r)]
        for a, b in partitions:
            if a not in d or b not in d:
                continue
            conf = c / d[a]
            lift = conf / (d[b] / n)
            if conf >= min_confidence and lift >= min_lift:
                lignes.append((tuple(sorted(a)), tuple(sorted(b)), c, c / n, conf, lift))
    cols = ["antecedent", "consequent", "support_abs", "support_rel", "confiance", "lift"]
    return pd.DataFrame(lignes, columns=cols).sort_values(["lift", "confiance"], ascending=False).reset_index(drop=True)


# ======================================================================
# Étape 4 : comprendre un pattern (profil d'un sous-groupe)
# ======================================================================
def masque_itemset(transactions: Sequence[FrozenSet[str]], itemset: Iterable[str],
                   index: Optional[pd.Index] = None) -> pd.Series:
    """Booléen par ligne : la transaction contient-elle l'itemset ? (aligné sur `index` si donné)."""
    x = frozenset(itemset)
    return pd.Series([x <= t for t in transactions], index=index)


def profiler_groupe(
    df: pd.DataFrame,
    masque: Union[pd.Series, np.ndarray],
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Décrit le sous-groupe `masque` colonne par colonne (TOUTES les colonnes, identifiants inclus,
    même ceux exclus des transactions : c'est ainsi qu'on trouve, par exemple, que toutes les
    lignes d'un pattern passent par le même transporteur).

      - colonnes catégorielles/identifiants : valeur dominante, sa part dans le groupe et dans
        l'ensemble ; un grand écart (concentration) signale une colonne qui caractérise le groupe ;
      - colonnes numériques / dates : intervalle et médiane du groupe vs de l'ensemble.
    Trié par concentration décroissante.
    """
    m = np.asarray(masque, dtype=bool)
    if m.sum() == 0:
        raise ValueError("Le groupe est vide.")
    lignes = []
    for c in (columns if columns is not None else df.columns):
        nature, s = _nature_colonne(df[c])
        if nature == "vide":
            continue
        if nature == "numérique":
            g, t = s[m].dropna(), s.dropna()
            if g.empty:
                continue
            lignes.append({"colonne": c, "nature": nature, "valeur_dominante": f"{g.min():g} .. {g.max():g} (méd. {g.median():g})",
                           "part_groupe": np.nan, "part_ensemble": np.nan, "n_modalites_groupe": g.nunique(),
                           "concentration": np.nan, "ensemble": f"{t.min():g} .. {t.max():g} (méd. {t.median():g})"})
        elif nature == "date":
            g, t = s[m].dropna(), s.dropna()
            if g.empty:
                continue
            lignes.append({"colonne": c, "nature": nature, "valeur_dominante": f"{g.min():%Y-%m-%d} .. {g.max():%Y-%m-%d}",
                           "part_groupe": np.nan, "part_ensemble": np.nan, "n_modalites_groupe": np.nan,
                           "concentration": np.nan, "ensemble": f"{t.min():%Y-%m-%d} .. {t.max():%Y-%m-%d}"})
        else:
            t = _normaliser_texte(s)
            g = t[m].dropna()
            if g.empty:
                continue
            vc = g.value_counts(normalize=True)
            top = vc.index[0]
            part_ens = float((t.dropna() == top).mean())
            lignes.append({"colonne": c, "nature": nature, "valeur_dominante": str(top), "part_groupe": float(vc.iloc[0]),
                           "part_ensemble": part_ens, "n_modalites_groupe": int(g.nunique()),
                           "concentration": float(vc.iloc[0]) - part_ens, "ensemble": ""})
    out = pd.DataFrame(lignes)
    return out.sort_values("concentration", ascending=False, na_position="last").reset_index(drop=True)


# ======================================================================
# Pipeline complet
# ======================================================================
def detecter_patterns(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    exclure: Iterable[str] = (),
    min_support: float = 0.05,
    min_abs: Optional[int] = None,
    max_len: Optional[int] = None,
    max_modalites: int = 30,
    n_classes: int = 3,
    dates: Sequence[str] = ("weekend",),
    min_lift: float = 3.0,
    min_abs_lift: int = 10,
    top: int = 8,
    regles: bool = False,
    min_confidence: float = 0.7,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Pipeline : DataFrame -> transactions -> itemsets fréquents -> patterns surprenants (lift).
    Renvoie {"transactions", "rapport", "itemsets", "fermes", "surprenants", ["regles"]}.

    Conseil : les itemsets les plus fréquents décrivent surtout la majorité. Les patterns utiles
    sont souvent RARES : baissez min_support (ex. 0.02) et lisez la table « surprenants »
    (lift élevé, support absolu suffisant).
    """
    transactions, rapport = construire_transactions(
        df, columns, exclure, max_modalites, n_classes, dates, verbose=show)
    F = itemsets_frequents(transactions, min_support, min_abs, max_len)
    fermes = filtrer_fermes_maximaux(F, "ferme")
    surpr = itemsets_surprenants(fermes, min_lift, min_abs_lift)
    res = {"transactions": transactions, "rapport": rapport, "itemsets": F, "fermes": fermes, "surprenants": surpr}
    if regles:
        res["regles"] = regles_association(F, min_confidence=min_confidence)

    if show:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 200)
        n = F.attrs["N"]
        seuil = min_abs if min_abs is not None else math.ceil(min_support * n)
        print(f"\nmin_support = {min_support if min_abs is None else min_abs / n:.3f} (support absolu >= {seuil}) "
              f"-> {len(F)} itemsets fréquents, dont {len(fermes)} fermés")
        print("Répartition par taille :", F.taille.value_counts().sort_index().to_dict())

        def _txt(t):
            return " ∧ ".join(t)

        print(f"\n--- Itemsets les plus FRÉQUENTS (taille >= 2) : décrivent surtout la majorité ---")
        d = F[F.taille >= 2].sort_values("support_abs", ascending=False).head(top)
        print(d.assign(itemset=d.itemset.map(_txt))[["itemset", "support_abs", "support_rel", "lift"]].round(3).to_string(index=False))
        print(f"\n--- Itemsets les plus SURPRENANTS (fermés, lift >= {min_lift}, support abs >= {min_abs_lift}) ---")
        if surpr.empty:
            print("  aucun : essayez de baisser min_support, min_lift ou min_abs_lift")
        else:
            d = surpr.head(top)
            print(d.assign(itemset=d.itemset.map(_txt))[["itemset", "taille", "support_abs", "support_rel", "lift"]].round(3).to_string(index=False))
        if regles and not res["regles"].empty:
            print(f"\n--- Règles d'association (confiance >= {min_confidence}) ---")
            r = res["regles"].head(top)
            print(r.assign(antecedent=r.antecedent.map(_txt), consequent=r.consequent.map(_txt)).round(3).to_string(index=False))
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for nom in ("itemsets", "fermes", "surprenants", "regles"):
            if nom in res:
                res[nom].assign(**{c: res[nom][c].map(" ∧ ".join) for c in ("itemset", "antecedent", "consequent")
                                   if c in res[nom].columns}).to_csv(Path(save_dir) / f"{nom}.csv", index=False)
    return res


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Détection de patterns (itemsets fréquents) sur un fichier CSV.")
    p.add_argument("fichier")
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--min-lift", type=float, default=3.0)
    p.add_argument("--min-abs-lift", type=int, default=10)
    p.add_argument("--dates", default="weekend", help="dérivées de dates, séparées par des virgules")
    p.add_argument("--exclure", default="", help="colonnes à ignorer, séparées par des virgules")
    p.add_argument("--regles", action="store_true")
    p.add_argument("--sortie", default=None, help="dossier où enregistrer les CSV de résultats")
    a = p.parse_args()
    detecter_patterns(
        lire_csv(a.fichier), exclure=[c for c in a.exclure.split(",") if c], min_support=a.min_support,
        max_len=a.max_len, dates=tuple(a.dates.split(",")), min_lift=a.min_lift,
        min_abs_lift=a.min_abs_lift, regles=a.regles, save_dir=a.sortie)