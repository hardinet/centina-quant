# CENTINA OMNI-QUANT v5.0 - Makefile
.PHONY: help install test test-fast lint format backtest sim dashboard paper live hyperopt stress clean docker-up docker-down docker-logs

PYTHON := python
PIP    := pip
SRC    := src

help:
	@echo "CENTINA OMNI-QUANT v5.0 - Available commands:"
	@echo ""
	@echo "  make install       Install all dependencies"
	@echo "  make test          Run unit test suite"
	@echo "  make test-fast     Run unit tests quickly"
	@echo "  make lint          Run flake8 + mypy"
	@echo "  make format        Run black + isort"
	@echo "  make backtest      Quick synthetic backtest"
	@echo "  make sim           Full historical simulation"
	@echo "  make dashboard     Launch Streamlit dashboard"
	@echo "  make paper         Start paper trading mode"
	@echo "  make live          Start AUTO mode"
	@echo "  make clean         Remove Python caches"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short -x

lint:
	$(PYTHON) -m flake8 $(SRC) --max-line-length=100 --ignore=E501,W503
	$(PYTHON) -m mypy $(SRC) --ignore-missing-imports --no-strict-optional

format:
	$(PYTHON) -m black $(SRC) tests scripts --line-length=100
	$(PYTHON) -m isort $(SRC) tests scripts

backtest:
	$(PYTHON) scripts/quick_backtest.py --strategy B --days 365

sim:
	$(PYTHON) scripts/run_historical_simulation.py --symbol BTCUSDT --strategy all --years 5

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

paper:
	$(PYTHON) -m src.main --mode PAPER

live:
	$(PYTHON) -m src.main --mode AUTO

hyperopt:
	$(PYTHON) -c "from src.learning.hyperoptimizer import HyperOptimizer; print('Hyperopt module ready')"

stress:
	$(PYTHON) -c "from src.historical_simulation.backtest.stress_test_engine import StressTestEngine; e = StressTestEngine(); print('Stress test engine ready')"

clean:
	$(PYTHON) scripts/clean.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f centina
