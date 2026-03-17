# 🚀 MLOps Project Template

> Production-ready MLOps template integrating the full ML lifecycle: from data versioning to model monitoring.

## 🏗️ Stack

| Tool | Usage |
|------|-------|
| 🐳 **Docker** | Containerization & orchestration |
| 🔬 **MLflow** | Experiment tracking & model registry |
| 🌀 **Airflow** | Pipeline orchestration & scheduling |
| 📦 **DVC** | Data & model versioning |
| 📊 **Prometheus + Grafana** | Infrastructure & performance monitoring |
| 🔍 **Evidently** | Data drift & model monitoring |
| ☸️ **Kubernetes** | Production deployment & scaling |

## 📁 Project Structure

```
mlops-template/
├── .github/                    # CI/CD workflows & issue templates
├── airflow/                    # DAGs & Airflow configuration
├── configs/                    # Environment configurations
├── data/                       # Data (versioned with DVC)
│   ├── raw/                    # Raw data (immutable)
│   ├── processed/              # Processed data
│   └── external/               # External data sources
├── docker/                     # Dockerfiles per service
├── docs/                       # Documentation
├── dvc/                        # DVC configuration & pipelines
├── kubernetes/                 # K8s manifests
├── mlflow/                     # MLflow configuration
├── monitoring/                 # Prometheus, Grafana, Evidently
├── notebooks/                  # Jupyter notebooks (exploration)
├── scripts/                    # Utility scripts
├── src/                        # Source code
│   ├── data/                   # Data ingestion & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Training, evaluation, inference
│   ├── monitoring/             # Monitoring & alerting logic
│   └── pipelines/              # Pipeline definitions
└── tests/                      # Unit & integration tests
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- kubectl (for Kubernetes)

### Local Setup

```bash
# Clone the template
git clone https://github.com/YOUR_USERNAME/mlops-template.git
cd mlops-template

# Setup Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc remote add -d myremote YOUR_REMOTE_STORAGE

# Start all services
docker-compose up -d

# Access the UIs
# MLflow:    http://localhost:5000
# Airflow:   http://localhost:8080
# Grafana:   http://localhost:3000
# Prometheus:http://localhost:9090
```

## 📋 Workflow

1. **Data** → Version with DVC, ingest via `src/data/`
2. **Features** → Engineer features via `src/features/`
3. **Training** → Train & track experiments with MLflow
4. **Orchestration** → Schedule pipelines with Airflow
5. **Monitoring** → Monitor drift with Evidently, infra with Prometheus/Grafana
6. **Deployment** → Deploy to Kubernetes via CI/CD

## 📚 Documentation

See [`docs/`](docs/) for detailed guides on each component.

## 📄 License

MIT License — see [LICENSE](LICENSE)
