# Année 2026-2027 - M2IA/M2DS - UE Data Mining - Projet Groupe 9

Sur ce repository, vous pourrez trouver l’ensemble des éléments
nécessaire au projet de groupe de l’UE Data Mining pour l’année 2026-2027.

Votre objectif est de mettre en évidence le narratif caché à l’intérieur
de ces données à l’aide des notions et méthodes vues au fur et
à mesure de l’UE Data Mining.

/!\ Ce narratif est entièrement fictif. Il a été créé dans un objectif pédagogique et ne reflète aucunement la réalité /!\

Pour ce faire, vous travaillerez en groupe en débattant et
en vous répartissant les tâches à faire.

Le rendu attendu est un rapport, sous la forme d’un notebook jupyter,
qui détaillera vos réflexions, les méthodes et vos conclusions.

Pour cela, le fichier `Base_notebook_to_complete.ipynb` est mis à votre disposition (vous pouvez changer le nom si vous le souhaitez).

Pensez à bien y préciser:

* Le nom de votre groupe
* La liste des membres

Vous serez avant tout évalué sur votre méthode, donc soyez précis, détaillé
et rigoureux dans vos descriptions.

/!\ L’usage de l’IA Générative est fortement déconseillé (ce sont vos cerveaux que l’on souhaite entrainer ici).
Si vous souhaitez tout de même l’utiliser, cela doit être justifié et documenté dans votre rapport /!\

## Jeux de données

Dans le dossier `data/`, vous trouverez les ensembles des jeux de données suivants:

- `data/public_records/` — jeux de données issus d’institutions administratives publiques,
- `data/health/` — jeux de données médicales issus de cliniques et d’hôpitaux,
- `data/ecology/` — jeux de données issus de recherches environnementales,
- `data/logistics/` — jeux de données issus de groupes et d’entreprises privées.

Le fichier `data/data_dictionary.csv` détaille les caractéristiques des variables présentes dans chaque jeu de données.

/!\ Ces jeux de données sont entièrement synthétiques et ont été créé dans un but pédogagique.
Ils ne représentent pas des informations factuelles à propos de personnes réelles,
d’organisations réelles, de gouvernnements réels ou localisations réelles.
Toute ressemblance avec des personnes réelles, physiques ou morales, est totalement involontaire et fortuite. /!\ 
 
## Pour bien commencer

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour gérer l’environnement Python
et les dépendances (`pyproject.toml` + `uv.lock`).

1. Faites un `fork` de ce repository vers un nouveau repository pour votre groupe.
2. Clonez le repository de votre groupe sur votre machine.
3. Installez `uv` si ce n’est pas déjà fait:
   * Linux / macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   * Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
4. Installez l’environnement et les dépendances avec la commande: `uv sync`
   (uv télécharge automatiquement la version de Python indiquée dans `.python-version`
   et crée le virtualenv dans `.venv/`)
5. Lancez le notebook avec la commande: `uv run jupyter notebook nom-du-notebook.ipynb`
   (ou `uv run jupyter lab`)
6. Travaillez en groupe et complétez le notebook

### Commandes utiles

* Ajouter une dépendance: `uv add nom-du-package`
* Retirer une dépendance: `uv remove nom-du-package`
* Exécuter une commande dans l’environnement: `uv run <commande>`
* Mettre à jour le lockfile: `uv lock --upgrade`

Le fichier `uv.lock` est versionné: il garantit que tous les membres du groupe
travaillent avec exactement les mêmes versions de paquets. Pensez à le commiter
après tout `uv add` / `uv remove`.

## Rendu

Le rendu pourra se faire via:

* une `pull-request`, 
* en m’envoyant une archive .zip contenant votre travail à `antoine.richard@chu-lyon.fr`.
