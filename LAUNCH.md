# Guide De Lancement

## Demarrage Recommande

```bash
python scripts/start.py --mode PAPER
```

Le lanceur demarre le dashboard officiel `dashboard/app.py` sur `http://localhost:8501`, puis garde l'agent dans le terminal pour les commandes interactives.

## Agent Seul

```bash
python -m src.main --mode PAPER
python -m src.main --mode ADVISOR
python -m src.main --mode SEMI
python -m src.main --mode AUTO
```

## Dashboard Seul

```bash
streamlit run dashboard/app.py --server.port 8501
```

`dashboard/web_interface.py` et `dashboard/centina_web.py` sont des variantes legacy conservees pour reference.

## Nouvelle Interface Web

Interface HTML/CSS/JavaScript servie par FastAPI, sans build Node:

```bash
python scripts/start.py --mode PAPER --ui web --port 8600
```

Puis ouvrir `http://localhost:8600`.

L'interface lit `data/bridge_state.json` et `data/cortex.db`, puis envoie les actions a l'agent via `data/cmd_queue.json`.

## Backtest

```bash
python scripts/quick_backtest.py --strategy B --days 365
python scripts/run_historical_simulation.py --symbol BTCUSDT --strategy all --years 5
```

## Prerequis

```bash
python -m pip install -r requirements.txt
```

Si Windows lance le mauvais Python, utiliser explicitement:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\start.py --mode PAPER
```

## Configuration Minimale

Copier `.env.example` vers `.env` et remplir:

```env
BINANCE_TESTNET=True
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
CAPITAL_USDT=200
DEFAULT_MODE=PAPER
ALLOW_LIVE_AUTO=False
```

Le live ne doit etre active qu'apres validation du paper gate et controle manuel.
