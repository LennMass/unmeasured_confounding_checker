# Multi-stage build: shared base, then API and dashboard targets.

FROM python:3.12-slim AS base
WORKDIR /app

# System deps: libgomp1 is the OpenMP runtime XGBoost needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e ".[dashboard]"

# ---- API target ----
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Dashboard target ----
FROM base AS dashboard
COPY dashboard ./dashboard
EXPOSE 8501
ENV API_URL=http://api:8000
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]