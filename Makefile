.PHONY: install api dashboard mlflow test docker-up docker-down

install:
	pip install -e ".[all]"

api:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

mlflow:
	mlflow ui --port 5001

test:
	python -m pytest tests/ -v

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down