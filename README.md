# CENTINA OMNI-QUANT v5.0

[![CI](https://github.com/hardinet/centina-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/hardinet/centina-quant/actions/workflows/ci.yml)
[![Pages](https://github.com/hardinet/centina-quant/actions/workflows/pages.yml/badge.svg)](https://github.com/hardinet/centina-quant/actions/workflows/pages.yml)

Agent de trading crypto Binance USDT en mode `PAPER`, `ADVISOR`, `SEMI` ou `AUTO`.

Auteur unique: ardin etienne.

> Projet portfolio technique. CENTINA ne constitue pas un conseil financier. Le mode recommande est `PAPER` avec `BINANCE_TESTNET=True`.

## Apercu

CENTINA est une base d'agent quant crypto orientee demonstration produit:

- agents specialises: Scout, Analyst, Historian, Guardian, Sentinel;
- scoring multi-strategies avec OCO, Kelly sizing, regime et momentum;
- risk engine avec circuit breaker, audit log et verrou live;
- dashboard Streamlit officiel et interface web FastAPI sans build Node;
- backtests, stress tests, donnees synthetiques et validation statistique;
- CI GitHub, GitHub Pages, configuration sans secrets et tests unitaires.

Site vitrine:

```text
https://hardinet.github.io/centina-quant/
```

Depot portfolio miroir:

```text
https://github.com/hardinet/centina-quant-portfolio
```

## Preuves

Commandes validees localement:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
.\.venv\Scripts\python.exe -m compileall -q src dashboard scripts simulation
.\.venv\Scripts\python.exe scripts\quick_backtest.py --strategy B --days 30
```

Etat attendu:

- `138+` tests unitaires passent.
- Le dashboard FastAPI repond sur `/` et `/api/health`.
- Le backtest rapide fonctionne avec donnees synthetiques.

## Demarrage Rapide

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
python scripts/start.py --mode PAPER
```

Dashboard web moderne:

```powershell
python scripts/start.py --mode PAPER --ui web --port 8600
```

Puis ouvrir:

```text
http://localhost:8600
```

Agent seul:

```powershell
python -m src.main --mode PAPER
```

Tests:

```powershell
python -m pytest tests/unit -q
```

## Configuration

Copier `.env.example` vers `.env`, puis remplir seulement les cles utiles:

```env
BINANCE_TESTNET=True
BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=
CAPITAL_USDT=200
DEFAULT_MODE=PAPER
ALLOW_LIVE_AUTO=False
```

Le live reste verrouille par `ALLOW_LIVE_AUTO=False` et par les controles du projet. Ne pas utiliser de vraies cles API dans un depot public.

## Architecture

```text
src/
  agents/                  Scout, Analyst, Historian, Guardian, Sentinel
  brain/                   scoring, strategies, OCO, Kelly, regime, satellites
  connectors/              Binance REST/WS, n8n webhook
  core/                    orchestrator, event bus, lifecycle
  database/                CortexDB SQLite
  historical_simulation/   backtests, validators, reports, synthetic data
  interface/               bridge dashboard + terminal helpers
  learning/                pattern library, optimizers, performance tracker
  notifications/           Apprise notifier
  security/                circuit breaker, audit logger, manipulation detector
  voice/                   optional voice interface

dashboard/
  app.py                   dashboard Streamlit officiel
  server.py                API FastAPI + interface web statique
  web/                     HTML/CSS/JS sans build Node

docs/
  index.html               vitrine GitHub Pages

scripts/
  start.py                 lance agent + dashboard
  quick_backtest.py        backtest rapide synthetique
  run_historical_simulation.py
```

## Modes

| Mode | Role |
| --- | --- |
| `PAPER` | Simulation, aucun ordre reel |
| `ADVISOR` | Propose les trades et attend `GO` / `PASSE` |
| `SEMI` | Execute seulement les signaux PRIME |
| `AUTO` | Execute automatiquement les signaux valides si les verrous le permettent |

## Commandes Interactives

| Commande | Action |
| --- | --- |
| `GO` | Execute la prochaine opportunite |
| `GO BTCUSDT` | Execute un symbole precis |
| `PASSE` | Ignore l'opportunite courante |
| `STATUT` | Positions, PnL, circuit breaker |
| `RAPPORT` | Rapport quotidien |
| `OPPS` | Opportunites detectees |
| `STRATS` | Strategies A/B/C/D/E |
| `HIST` | Derniers trades |
| `AUTO` | Bascule auto |
| `PAPER` | Bascule simulation |
| `VOICE` | Active/desactive la voix |
| `BACKTEST` | Backtest rapide |
| `OPTIMISE` | Optimisation Optuna |
| `AIDE` | Aide |
| `EXIT` | Arret propre |

## Strategies

| ID | Nom | Usage |
| --- | --- | --- |
| A | Breakout Momentum | Breakout de resistance + fort volume |
| B | Golden Cross | EMA7 au-dessus EMA25 en trend normal |
| C | News Momentum | News bullish + RSI momentum |
| D | Volume Profile | Scalp en marche ranging |
| E | Liquidity Reversal | Reversal apres wick/liquidity hunt |

Les definitions runtime sont dans `src/brain/strategy_selector.py`.

## Publication

Le site statique est dans `docs/index.html` et peut etre publie avec GitHub Pages via le workflow:

```text
.github/workflows/pages.yml
```

Dans GitHub, configurer `Settings > Pages > GitHub Actions`, puis lancer `Deploy GitHub Pages`.
