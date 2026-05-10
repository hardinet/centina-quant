# CENTINA OMNI-QUANT v5.0

Agent de trading crypto Binance USDT en mode paper, advisor, semi-auto ou auto.

## Portfolio Recruteur

Auteur unique: ardin etienne.

Ce depot est prepare pour etre partage comme projet portfolio:

- Vitrine statique GitHub Pages: `docs/index.html`
- Interface web moderne: `dashboard/web/`
- API dashboard FastAPI: `dashboard/server.py`
- Tests unitaires: `python -m pytest tests/unit -q`
- CI GitHub: `.github/workflows/ci.yml`
- Guide de publication du second depot: `GITHUB_PUBLICATION.md`

Le projet est une demonstration technique et ne constitue pas un conseil financier.

Le projet a maintenant une route officielle simple:

- Lanceur complet: `python scripts/start.py --mode PAPER`
- Agent seul: `python -m src.main --mode PAPER`
- Dashboard officiel: `streamlit run dashboard/app.py`
- Dashboard web moderne: `python scripts/start.py --mode PAPER --ui web --port 8600`
- Tests: `python -m pytest tests/unit -q`

> Prudence: ce projet manipule des signaux de trading. Commencer en `PAPER`, garder `BINANCE_TESTNET=True`, et ne jamais activer le live sans audit.

## Structure

```text
src/
  agents/                  Scout, Analyst, Historian, Guardian, Sentinel
  brain/                   scoring, strategies, OCO, Kelly, regime, satellites
  connectors/              Binance REST/WS, n8n webhook
  core/                    orchestrator, event bus, lifecycle
  database/                CortexDB SQLite
  historical_simulation/   backtests, validators, reports, synthetic data
  interface/               bridge dashboard + terminal helpers
  learning/                pattern library, backtests, optimizers
  notifications/           Apprise notifier
  security/                circuit breaker, audit logger, manipulation detector
  voice/                   optional voice interface

dashboard/
  app.py                   dashboard Streamlit officiel
  server.py                dashboard FastAPI + HTML/CSS/JS optionnel
  web/                     interface graphique moderne sans build Node
  web_interface.py         ancien dashboard simple
  centina_web.py           ancien dashboard pro

config/
  centina_config.yaml.example
  historical/
  strategies/

scripts/
  start.py                 lance agent + dashboard
  quick_backtest.py
  run_historical_simulation.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sur PowerShell, si `python` pointe vers le Microsoft Store ou refuse de s'executer, utiliser le chemin complet:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configuration

Copier `.env.example` vers `.env`, puis remplir les cles utiles:

```env
BINANCE_TESTNET=True
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
CAPITAL_USDT=200
DEFAULT_MODE=PAPER
ALLOW_LIVE_AUTO=False
```

Le fichier `config/centina_config.yaml.example` documente la structure applicative, mais le runtime actuel lit surtout les variables `.env`.

## Lancement

Lancement recommande:

```bash
python scripts/start.py --mode PAPER
```

Cela lance:

- le dashboard sur `http://localhost:8501`
- l'agent dans le terminal courant
- les logs dashboard dans `logs/streamlit.log`

Autres commandes:

```bash
python -m src.main --mode ADVISOR
python -m src.main --mode SEMI
python -m src.main --mode AUTO
streamlit run dashboard/app.py --server.port 8501
python scripts/start.py --mode PAPER --ui web --port 8600
python scripts/quick_backtest.py --strategy B --days 365
```

## Modes

| Mode | Role |
| --- | --- |
| `PAPER` | Simulation, aucun ordre reel |
| `ADVISOR` | Propose les trades et attend `GO` / `PASSE` |
| `SEMI` | Execute seulement les signaux PRIME |
| `AUTO` | Execute automatiquement les signaux valides |

Le live reste verrouille par le paper gate 7 jours et par `ALLOW_LIVE_AUTO`.

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

## Etat du Rangement

Cette version privilegie `dashboard/app.py` et `scripts/start.py`. Une interface web plus moderne existe aussi via `dashboard/server.py` et `dashboard/web/`; elle parle au meme agent par `data/bridge_state.json`, `data/cortex.db` et `data/cmd_queue.json`.
