# Publication GitHub Pour Recruteurs

Objectif: publier CENTINA dans un second depot GitHub public propre, lisible et
accessible aux recruteurs.

## 1. Creer le second depot

Nom conseille:

```powershell
centina-quant-portfolio
```

Avec GitHub CLI connecte:

```powershell
gh repo create centina-quant-portfolio --public --source=. --remote=origin --push
```

Sans GitHub CLI:

1. Creer un depot public vide sur GitHub.
2. Copier son URL HTTPS.
3. Lancer:

```powershell
git remote add origin https://github.com/hardinet/centina-quant-portfolio.git
git branch -M main
git push -u origin main
```

## 2. Activer la vitrine GitHub Pages

Le dossier `docs/` contient une page statique faite pour les recruteurs.

Dans GitHub:

1. Ouvrir `Settings`.
2. Aller dans `Pages`.
3. Choisir `GitHub Actions` comme source.
4. Lancer le workflow `Deploy GitHub Pages`.

La page publique ressemblera a:

```text
https://hardinet.github.io/centina-quant-portfolio/
```

## 3. Verifier avant partage

Commandes de controle:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
.\.venv\Scripts\python.exe -m compileall -q src dashboard scripts simulation
.\.venv\Scripts\python.exe scripts\quick_backtest.py --strategy B --days 30
```

## 4. Auteur

La vitrine mentionne un auteur unique: ardin etienne.
