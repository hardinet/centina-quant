# Dashboards

Official dashboard:

```bash
streamlit run dashboard/app.py --server.port 8501
```

Optional modern web dashboard:

```bash
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8600
```

Legacy variants kept for reference:

- `dashboard/web_interface.py`
- `dashboard/centina_web.py`

New UI work should target `dashboard/app.py` unless there is a specific reason
to revive a legacy variant.
