web: uvicorn app:app --host=0.0.0.0 --port=${PORT:-8000} --workers=1
worker: python worker.py  # If using Celery for async tasks