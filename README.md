# IMDB sentiment analysis — end-to-end MLOps project

This project trains a model that reads an IMDb review and decides whether it's positive or negative. That part is almost incidental. The actual point is everything wrapped around the model: data that's versioned instead of just sitting in a folder, experiments that are tracked instead of remembered, a registry that decides what's actually running in production, a CI/CD pipeline that won't let broken code ship, a Kubernetes deployment on AWS, and a monitoring stack that tells you when something's wrong before a user does.

Most tutorials stop at "train a model, save a pickle file, wrap it in a Flask app." This goes further, because that's where real systems actually break — not during training, but during the six months after deployment when data drifts, dependencies update, and nobody's watching.

## Architecture

```
Developer
   │  git push
   ▼
GitHub  ──────────────►  GitHub Actions (CI/CD)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Unit Tests       DVC Repro       Flask Tests
                              │
                              ▼
                    Model Training & Evaluation
                              │
                              ▼
                    MLflow + DagsHub (tracking)
                              │
                              ▼
                    Model Registry (staging → prod)
                              │
                              ▼
                          Docker Image
                              │
                              ▼
                           AWS ECR
                              │
                              ▼
                           AWS EKS
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
           Flask Pod 1                Flask Pod 2
                 │                         │
                 └────────────┬────────────┘
                               ▼
                     Kubernetes LoadBalancer
                               │
                               ▼
                       Sentiment web UI


        Monitoring (runs alongside, not after)
        ─────────────────────────────────────
        Flask Pods → /metrics → Prometheus → Grafana
                                    │
                        CPU, memory, request rate,
                        latency, error rate, uptime
```

The two halves matter separately. The top half is how a model goes from a Jupyter notebook to something serving live traffic. The bottom half is how you find out that traffic is actually being served correctly, which most people skip and then wonder why they only hear about outages from angry users.

## Model

Logistic regression, `C=1`, `solver=liblinear`, `penalty=l1`. Features come from Bag of Words — each review gets turned into a sparse vector of word counts, and the classifier learns which words push a review toward positive or negative. Nothing exotic. The complexity in this project isn't in the model architecture, it's in the plumbing around it, which is honestly where most of the engineering time in a real ML system goes anyway.

Input:
```
"This product is absolutely amazing!"
```
Output:
```
Positive
```

## Project structure

```
MLOPs-capstone-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── model.pkl
│
├── reports/
│   └── metrics.json
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train_model.py
│   │   ├── model_evaluation.py
│   │   └── register_model.py
│   ├── flask_app/
│   │   ├── app.py
│   │   └── templates/
│   └── logger/
│
├── tests/
│   ├── test_model.py
│   └── test_flask_app.py
│
├── scripts/
│   └── promote_model.py
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── secret.yaml
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── .env
└── README.md
```

## Data versioning and the DVC pipeline

The pipeline is broken into stages: ingestion, preprocessing, feature engineering, training, evaluation. Each stage declares what it depends on and what it produces, and DVC only reruns a stage if something upstream of it actually changed. That's the whole value proposition over a plain Python script — you get reproducibility without having to manually track which script produced which model.

```bash
dvc repro
```

runs whatever's stale. To see the stage graph:

```bash
dvc dag
```

The pipeline in order:

```
Data Ingestion
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
```

Because inputs and outputs are hashed and tracked, the same data plus the same `params.yaml` will always reproduce the same model. That sounds obvious until you've been on a team where "which data was this model trained on" takes an afternoon to answer.

## Evaluation

Four metrics get computed after training and written to `reports/metrics.json`:

- **Accuracy** — correct predictions over total predictions. Simple, but misleading if the classes are imbalanced.
- **Precision** — of everything the model called positive, how much actually was. Matters when a false positive is expensive.
- **Recall** — of everything that actually was positive, how much the model caught. Matters when missing a positive is expensive.
- **ROC-AUC** — how well the model separates the two classes across every possible threshold, not just the default 0.5 cutoff.

Example output:
```json
{
  "accuracy": 0.85,
  "precision": 0.84,
  "recall": 0.87,
  "auc": 0.91
}
```

## Experiment tracking with MLflow

Every training run logs its parameters, its metrics, and the model artifact itself, so nothing gets lost between runs and nobody has to keep a spreadsheet of "which run had the best recall."

```python
mlflow.sklearn.log_model(clf, name="model")
```

Tracked per run:
- Parameters — `C`, `solver`, `penalty`, `max_iter`
- Metrics — accuracy, precision, recall, auc
- The model artifact itself

## DagsHub as the remote

MLflow needs somewhere to actually store all this, and rather than self-host a tracking server, this project points MLflow at DagsHub:

```python
dagshub.init(
    repo_owner="anurag-23werty",
    repo_name="MLOPs-capstone-project",
    mlflow=True
)
```

That gets you a hosted MLflow UI, reachable from anywhere, without babysitting infrastructure just to look at a metrics dashboard.

## Model registry

Registered as `my_model`. Every training run that gets logged can become a new version:

```
my_model
 ├── Version 12
 ├── Version 13
 └── Version 14
```

Versions move through stages as they earn trust:

```
Registered
     ↓
Staging
     ↓
Production
```

The Flask app doesn't load a `.pkl` file off disk. It pulls whatever's currently marked production, by URI:

```
models:/my_model/14
```

That one decision — referencing the registry instead of a hardcoded path — is what makes promotion actually mean something. You can retrain, evaluate, and promote a new version without touching the app's code or redeploying anything except the model reference itself.

## Testing

Two test suites, run separately or together:

```bash
python -m unittest tests/test_model.py
python -m unittest tests/test_flask_app.py
python -m unittest discover
```

`test_model.py` checks the model behaves the way it should — loads correctly, predicts in the expected format, doesn't silently degrade. `test_flask_app.py` checks the API contract — the right routes exist, `/predict` returns what it's supposed to. CI runs both automatically, so a broken model or a broken endpoint fails the build before it ever reaches a container.

## Serving with Flask

```bash
python src/flask_app/app.py
```
or, for something closer to production:
```bash
gunicorn -w 2 -b 0.0.0.0:5000 src.flask_app.app:app
```

Two routes:
- `GET /` — the web UI for entering a review and seeing a prediction
- `POST /predict` — the actual inference endpoint, takes review text and returns positive or negative

Request flow once it's deployed:
```
Gunicorn → Flask → MLflow Model Registry → production model → response
```

## Docker

```bash
docker build -t flask-app .
docker run -p 5000:5000 --env-file .env flask-app
```

Then it's at `http://localhost:5000`. The container runs the app through Gunicorn rather than Flask's own dev server, since the dev server isn't meant to hold up under real traffic.

## AWS ECR

Images get pushed to Elastic Container Registry so EKS has somewhere to pull them from:

```bash
aws ecr get-login-password --region us-east-1 | docker login \
  --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

docker build -t flask-app:latest .
docker tag flask-app:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
```

CI does this automatically: build → authenticate → tag → push, on every merge that passes tests.

## Kubernetes and AWS EKS

The app runs on an actual EKS cluster, not a local minikube setup, because the whole point is demonstrating how this looks in a real cloud environment.

Creating the cluster:
```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 \
  --nodes-min 1 \
  --nodes-max 1 \
  --managed

aws eks update-kubeconfig --region us-east-1 --name flask-app-cluster
kubectl get nodes
```

Deploying:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
```

The deployment runs two replicas:
```yaml
replicas: 2
```

so the app looks like this once it's up:
```
                LoadBalancer
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
      Flask Pod 1          Flask Pod 2
```

If one pod crashes, Kubernetes brings it back to maintain the replica count on its own — nobody has to get paged for a single pod dying.

The service itself is a `LoadBalancer` type:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: flask-app-service
spec:
  type: LoadBalancer
  selector:
    app: flask-app
  ports:
    - port: 5000
      targetPort: 5000
```

AWS provisions the actual load balancer automatically once this is applied. To find where the app actually lives:
```bash
kubectl get svc flask-app-service
```

## Secrets

Nothing sensitive — DagsHub tokens, AWS keys — gets baked into the Docker image or committed to the repo. They're injected at runtime through Kubernetes secrets:

```bash
kubectl create secret generic capstone-secret \
  --from-literal=CAPSTONE_TEST="$CAPSTONE_TEST"
```

and referenced in the deployment like this:
```yaml
env:
  - name: CAPSTONE_TEST
    valueFrom:
      secretKeyRef:
        name: capstone-secret
        key: CAPSTONE_TEST
```

For GitHub Actions, the equivalent secrets — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REPOSITORY`, `TRACKING_URI`, `CAPSTONE_TEST` — live under the repo's Settings → Secrets and variables → Actions, never in the workflow file itself.

## Monitoring: Prometheus and Grafana

Deploying an app and walking away isn't really finished — you find out it's broken when a user complains, which is the slowest and worst possible way to find out. So the deployment ships with a monitoring stack sitting next to it, not bolted on afterward.

```
Flask Application
       │
       │  /metrics
       ▼
   Prometheus  ──►  Time-series DB
       │
       ▼
    Grafana  ──►  Dashboards
```

**Prometheus** scrapes `/metrics` on a schedule and stores everything as time series: request count, request rate, error rate, response latency, CPU usage, memory usage, pod availability, uptime. None of this requires touching application logic beyond exposing the metrics endpoint.

**Grafana** sits on top of Prometheus as the visualization layer. A typical dashboard for this app looks roughly like:

```
┌───────────────────────────────────────────┐
│           Flask Application                │
│              Monitoring                    │
├───────────────────┬───────────────────────┤
│ Request Rate       │ Error Rate            │
│    120 req/s        │     0.4%              │
├───────────────────┼───────────────────────┤
│ Response Latency   │ Pod Status             │
│     42 ms            │     2 / 2              │
├───────────────────┼───────────────────────┤
│ CPU Usage          │ Memory Usage           │
│     34%              │     47%                │
└───────────────────┴───────────────────────┘
```

The full path from pod to dashboard:
```
                    AWS EKS
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Flask Pod 1         Flask Pod 2
             │                   │
             └─────────┬─────────┘
                        ▼
                   /metrics
                        │
                        ▼
                  Prometheus
                        │
                        ▼
                   Grafana
                        │
                        ▼
             Monitoring dashboard
```

What this actually catches, in practice:

- **Latency creep** — normal is around 50ms; if it drifts up toward 900ms, something's degrading before it becomes a full outage.
- **Rising error rate** — a healthy service sits near 98% HTTP 200s; watching the 500s tick up gives you a warning before users start complaining en masse.
- **Pod failures** — desired replicas is 2, and if available drops to 1, you know before the load balancer starts struggling to keep up.
- **Resource pressure** — CPU past 80% or memory past 90% usually means it's time to scale, not wait for a crash.

This is the difference between deployment being the finish line and deployment being one step in an ongoing feedback loop.

## CI/CD with GitHub Actions

Every push triggers the pipeline, and nothing skips a step:

```
git push
   ↓
Checkout
   ↓
Setup Python
   ↓
Install dependencies
   ↓
DVC repro
   ↓
Model tests
   ↓
Model evaluation
   ↓
Model promotion
   ↓
Flask tests
   ↓
Docker build
   ↓
AWS ECR login
   ↓
Docker push
```

If model tests fail, nothing downstream runs — no promotion, no build, no push. The workflow lives in `.github/workflows/ci.yml`. The point of this ordering isn't ceremony, it's that a bad model or a broken endpoint gets caught in CI instead of in production, which is a much cheaper place to catch it.

## Local development

```bash
git clone <YOUR_GITHUB_REPOSITORY>
cd MLOPs-capstone-project

python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

`.env` file:
```
TRACKING_URI=<YOUR_MLFLOW_TRACKING_URI>
CAPSTONE_TEST=<YOUR_DAGSHUB_TOKEN>
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY>
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET>
AWS_DEFAULT_REGION=us-east-1
```

This should never be committed — it's already covered by `.gitignore`.

Then:
```bash
dvc repro
python -m unittest discover
python src/flask_app/app.py
```

## Useful Kubernetes commands

```bash
kubectl get nodes
kubectl get pods
kubectl get svc
kubectl describe deployment flask-app
kubectl describe svc flask-app-service
kubectl logs <pod-name>
kubectl exec -it <pod-name> -- /bin/sh
kubectl rollout restart deployment flask-app
kubectl rollout status deployment flask-app
kubectl delete deployment flask-app
kubectl delete service flask-app-service
```

## Cloud cost management

This EKS setup is meant for demonstration, not to sit running indefinitely. Cloud infrastructure costs money whether or not it's serving traffic, so once a demo is done, tearing the cluster down is the sane default:

```bash
eksctl delete cluster \
  --name flask-app-cluster \
  --region us-east-1
```

Everything that actually matters for reproducibility — source code, Dockerfile, Kubernetes manifests, the CI/CD workflow, MLflow experiment history, monitoring config — lives outside the cluster and survives the teardown. Spinning the cluster back up when it's needed again is a matter of rerunning `eksctl create cluster` and the deploy steps above, not rebuilding anything from scratch.

## Demonstrating the project

A reasonable walkthrough, in order:

1. GitHub repository and structure
2. DVC pipeline (`dvc dag`)
3. MLflow experiment runs on DagsHub
4. The registered model and its versions
5. GitHub Actions CI/CD run
6. Docker image sitting in ECR
7. The EKS cluster and its nodes
8. Running pods (`kubectl get pods`)
9. The Flask app, live
10. An actual sentiment prediction
11. Prometheus metrics
12. The Grafana dashboard

A short screen recording covering this end to end (two to five minutes is plenty) is a lot more convincing than a wall of text, and it means you don't have to keep AWS infrastructure running just so someone can look at it later.

## Tech stack

| Category | Technology |
|---|---|
| Programming | Python |
| ML | Scikit-learn |
| Data processing | Pandas, NumPy |
| Data versioning | DVC |
| Experiment tracking | MLflow |
| ML platform | DagsHub |
| Model registry | MLflow Model Registry |
| Testing | Python unittest |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Container registry | AWS ECR |
| Orchestration | Kubernetes |
| Cloud Kubernetes | AWS EKS |
| Load balancing | AWS LoadBalancer |
| Monitoring | Prometheus |
| Visualization | Grafana |
| Application | Flask |
| Production server | Gunicorn |
| Cloud | AWS |
| Environment management | python-dotenv |

## What this demonstrates

Data versioning, reproducible pipelines, experiment tracking, model evaluation, a model registry with real promotion semantics, automated testing, CI, CD, Docker, a container registry, Kubernetes deployments and services, AWS EKS and ECR, load balancing, secrets management, and application monitoring with Prometheus and Grafana feeding real dashboards.

## Roadmap

Terraform for infrastructure instead of manual `eksctl` calls. Helm charts instead of raw manifests. A Horizontal Pod Autoscaler so replica count reacts to load automatically. Readiness and liveness probes so Kubernetes actually knows when a pod is unhealthy versus just slow. Prometheus Alertmanager so monitoring pages someone instead of just sitting on a dashboard. Automated Grafana dashboard provisioning. Distributed tracing with OpenTelemetry. Model and data drift detection. Automated retraining triggered by that drift. Canary and blue-green deployments instead of a straight rollout. GitOps through ArgoCD. AWS CloudWatch as a second monitoring layer. TLS, a real domain, API authentication, and rate limiting.

## Lessons learned

Training a model and saving a `.pkl` file is maybe ten percent of what it takes to run one in production. The rest is reliable data, reproducible training, experiment tracking so nothing gets lost, model versioning so promotion actually means something, automated testing so bad code doesn't ship, CI/CD so shipping is boring instead of terrifying, containerization and orchestration so it actually stays up, and monitoring so you find out about problems from a dashboard instead of a user's angry email. The goal here was never just a good accuracy number — it was proving the whole path from a notebook to a system somebody could actually depend on.

## Author

Anurag — CSE (AI/ML)
