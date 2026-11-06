.PHONY: setup train test lint docker-build drift-report dvc-init

setup:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

dvc-init:
	dvc init
	dvc remote add -d myremote s3://your-bucket/dvc-store

train:
	dvc repro

test:
	pytest tests/ -v --cov=src

lint:
	black src/ tests/
	flake8 src/ tests/
	isort src/ tests/

docker-build:
	docker build -f docker/Dockerfile.train -t mlops-train:latest .
	docker build -f docker/Dockerfile.serve -t mlops-serve:latest .

drift-report:
	python monitoring/evidently/drift_report.py

stack-up:
	docker-compose up -d

stack-down:
	docker-compose down
