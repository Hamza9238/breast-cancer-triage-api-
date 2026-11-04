# Breast Cancer Triage Service

A containerized ML service that classifies cell-nucleus measurements from fine needle aspirate samples as **malignant** or **benign**, helping pathologists prioritize their review queue.

> **Full Assignment Report**: See [REPORT.md](REPORT.md) for the detailed technical report, cross-validation benchmarks, quality gate verification, and architecture screenshots.

## Repository Structure

```
breast-cancer-triage-service/
├── .dvc/                        # DVC configuration
├── app/                         # FastAPI service
│   ├── __init__.py
│   ├── main.py                  # API endpoints
│   └── schemas.py               # Pydantic models
├── training/                    # Training pipeline
│   ├── __init__.py
│   ├── prepare_data.py          # Dataset export script
│   └── train.py                 # Model training script
├── data/raw/                    # DVC-tracked dataset
│   └── breast_cancer.csv.dvc    # DVC pointer file
├── dvc-remote/                  # Local DVC remote (committed for reproducibility)
├── artifacts/                   # Model artifacts (DVC-tracked, not in Git)
├── Dockerfile                   # Production multi-stage image
├── Dockerfile.naive             # Baseline naive image
├── docker-compose.yml           # Full stack (build + redis)
├── docker-compose.hub.yml       # Clean-room stack (pull from Hub)
├── .env.example                 # Environment variable template
├── params.yaml                  # Training hyperparameters
├── dvc.yaml                     # DVC pipeline definition
├── dvc.lock                     # DVC pipeline lock file
├── requirements-train.txt       # Training dependencies
├── requirements-serve.txt       # Serving dependencies
├── requirements-dev.txt         # Development dependencies
├── .gitignore
├── .dvcignore
├── .dockerignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (WSL 2 backend on Windows) or Docker Engine on Linux
- Docker Compose v2 (`docker compose` command)
- DVC (`pip install dvc`)

### 1. Clone and Restore Data

```bash
git clone https://github.com/Hamza9238/breast-cancer-triage-api-.git
cd breast-cancer-triage-api-

# Install development dependencies
pip install -r requirements-dev.txt

# Restore DVC-tracked data from the local remote
dvc pull
```

### 2. Run the Training Pipeline

```bash
# Run via DVC pipeline (recommended — handles caching)
dvc repro

# Or run directly
python training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
```

### 3. Run the API Locally (without Docker)

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

### 4. Build the Production Docker Image

```bash
# Build with version info
docker build \
  --build-arg MODEL_VERSION=1.0.0 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t hamzayousaf1/breast-cancer-triage-api:1.0.0 .
```

### 5. Run the Image Standalone

```bash
docker run -d -p 8000:8000 --name triage-api \
  hamzayousaf1/breast-cancer-triage-api:1.0.0
```

### 6. Start the Full Compose Stack

```bash
# Copy the example env and edit
cp .env.example .env
# Edit .env with your values

# Build and start
docker compose up -d --build

# Check health
docker compose ps

# View logs
docker compose logs -f api
```

### 7. Push to Docker Hub

```bash
# Login
docker login

# Tag and push
docker tag hamzayousaf1/breast-cancer-triage-api:1.0.0 hamzayousaf1/breast-cancer-triage-api:latest
docker push hamzayousaf1/breast-cancer-triage-api:1.0.0
docker push hamzayousaf1/breast-cancer-triage-api:latest
```

### 8. Clean-Room Verification

```bash
# Remove all local images
docker rmi hamzayousaf1/breast-cancer-triage-api:1.0.0
docker rmi hamzayousaf1/breast-cancer-triage-api:latest

# Create a clean directory with only:
#   - docker-compose.hub.yml
#   - .env (with your DOCKER_USERNAME=hamzayousaf1)
docker compose -f docker-compose.hub.yml up -d
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check (returns `{"status": "ok"}`) |
| `/predict` | POST | Classify one sample (30 features) |
| `/model/info` | GET | Model version, family, metrics, git SHA |
| `/stats` | GET | Prediction count (Redis-backed) |

### Sample Prediction Request

```bash
python -c "from sklearn.datasets import load_breast_cancer as l; import json; d=l(); print(json.dumps(dict(zip([n.replace(' ','_') for n in d.feature_names], map(float, d.data[0])))))" | curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @-
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `artifacts/model.joblib` | Path to the model artifact |
| `DECISION_THRESHOLD` | `0.5` | Malignant probability cutoff |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `REDIS_HOST` | *(empty)* | Redis hostname (enables counter) |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL_VERSION` | `1.0.0` | Model version string |
| `GIT_SHA` | `unknown` | Git commit hash |

## Docker Hub

- **Repository**: [hamzayousaf1/breast-cancer-triage-api](https://hub.docker.com/r/hamzayousaf1/breast-cancer-triage-api)
- **Tags**: `1.0.0`, `latest`
