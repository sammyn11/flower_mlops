# Flower Classification MLOps Project

## Project overview

I built an end-to-end image classification and MLOps pipeline for five flower classes: daisy, dandelion, rose, sunflower, and tulip. The project uses the TF Flowers image dataset and an ImageNet-pretrained MobileNetV2 model. It covers data acquisition and preprocessing, model training and evaluation, image prediction, monitoring, bulk uploads, model retraining, Docker deployment, and Locust load testing.

## Project links

- GitHub repository: https://github.com/sammyn11/flower_mlops
- Demonstration video: https://youtu.be/YFKE3cYcEh8
- Deployed dashboard: https://flower-mlops-dashboard.onrender.com
- Deployed API documentation: https://flower-mlops-api.onrender.com/docs
- Deployed API status: https://flower-mlops-api.onrender.com/status

## Pipeline and application

The notebook downloads the TF Flowers images, creates stratified training and test splits, explores the image data, and preprocesses it for training. I compare a small CNN baseline with a tuned MobileNetV2 transfer-learning model. The saved artifacts are `models/flower_model.h5`, `models/class_names.json`, and `models/model_meta.json`.

The FastAPI service exposes prediction, monitoring, dataset insight, upload, and training endpoints. The Streamlit dashboard provides single-image prediction, dataset visualizations, service/model status, and bulk image upload with a retraining trigger. Uploaded images are validated, converted to RGB, resized to 224 × 224, stored by label, and recorded in SQLite. Retraining loads the current saved model, fine-tunes it with uploaded images and a replay sample from the original training set, evaluates it, and saves a new model version.

The API endpoints are:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Predict a flower class from one uploaded image |
| `GET /status` | Show API uptime, model version, metrics, upload counts, and job status |
| `GET /metrics` | Return the saved model evaluation metadata |
| `GET /insights` | Return image dataset statistics for the dashboard |
| `POST /upload` | Validate and store a batch of labeled images |
| `POST /retrain` | Trigger fine-tuning on uploaded data |
| `POST /train` | Trigger initial training |
| `GET /job` | Return training or retraining progress and logs |

## Reproducing the project

### Notebook training and evaluation

`notebook/flower_mlops.ipynb` contains the detailed preprocessing, visualizations, baseline CNN, MobileNetV2 training and tuning, test evaluation, and prediction functions. It is designed to run in Google Colab with a GPU runtime. Colab provides TensorFlow; the notebook’s setup cell locates the repository under `/content`. If the other project dependencies are missing, they can be installed with:

```python
%pip install -r requirements-colab.txt
```

Running the notebook downloads and splits the image dataset, trains and evaluates both models, and saves the model artifacts under `models/`. The notebook reports accuracy, loss, macro precision, macro recall, macro F1, a per-class report, and a confusion matrix.

### Local Docker deployment

The local Docker Compose deployment runs the API, Nginx load balancer, and dashboard. The host data directory is mounted into the API containers, so the TF Flowers split must be present in `data/` before startup. The acquisition helper creates that split:

```bash
python -m pip install -r requirements-colab.txt
python scripts/download_data.py
docker compose up -d --build --scale api=2
```

The local dashboard is available at `http://localhost:8501`, the load-balanced API documentation at `http://localhost:8080/docs`, and API status at `http://localhost:8080/status`. The Compose configuration shares `data/` and `models/` across API replicas. The services can be stopped with `docker compose down`.

### Render deployment

I deployed two Docker web services to Render’s Frankfurt region from the `render.yaml` Blueprint. The API image includes the trained model and downloads the TF Flowers dataset during its image build. I verified the hosted dashboard and prediction flow at the links above.

Render’s free web services sleep after 15 minutes without traffic and can take about a minute to wake. Their filesystem is ephemeral: uploaded retraining files, the SQLite upload log, and retrained model changes can be lost after a restart, sleep, or redeploy. Free services also run only one instance. For these reasons, I demonstrate the interactive hosted prediction and dashboard on Render, and report the multi-container load test from the local Docker environment separately.

## Dataset visualizations

The dashboard presents class counts, mean image brightness, average RGB colour, and original image dimensions. The class-count chart shows whether examples are evenly represented and whether the classifier may favor a larger class. The brightness chart shows variation in lighting within each flower class, which motivates augmentation and can explain why brightness alone is not a reliable class cue. The RGB chart shows the contribution of petal colour and foliage; overlapping colour profiles help explain why visually similar flowers can be confused. Image dimensions and aspect ratios show how the original photos vary before they are resized to 224 × 224 for the model.

## Model evaluation

The following scores are from the held-out test set. The optimized MobileNetV2 model outperformed the scratch CNN baseline. After retraining with three uploaded images and a replay sample, model v3 reached 90.97% test accuracy. Its confusion matrix showed notable errors between daisies and dandelions, and between tulips and roses.

| Model | Accuracy | Loss | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.6772 | 0.8292 | 0.6909 | 0.6668 | 0.6722 |
| MobileNetV2 (optimized notebook model) | 0.9042 | 0.3121 | 0.9032 | 0.9060 | 0.9041 |
| MobileNetV2 (deployed model v3) | 0.9097 | 0.3007 | 0.9080 | 0.9109 | 0.9093 |

The deployed API status reports model v3, the saved test metrics, and the 731-image test-set size. These are held-out model evaluation results, not metrics calculated from live user requests.

## Locust flood-request simulation

I used Locust 2.29.1 to send image-prediction and status requests through Nginx. Each configuration ran for 60 seconds after the API replicas were warmed. I tested 50, 100, and 200 concurrent users with one, two, and three API containers. The client timeout was 15 seconds. Full results, including failure rate and latency, are in [`results/benchmark_summary.md`](results/benchmark_summary.md).

| API containers | Users | Requests | Failures | Failure rate | Average (ms) | Median (ms) | p95 (ms) | Attempts/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 395 | 0 | 0% | 5,290 | 6,700 | 7,600 | 6.67 |
| 1 | 100 | 383 | 158 | 41.25% | 11,278 | 15,000 | 15,000 | 6.45 |
| 1 | 200 | 652 | 531 | 81.44% | 13,867 | 15,000 | 15,000 | 11.05 |
| 2 | 50 | 551 | 0 | 0% | 3,561 | 4,300 | 5,300 | 9.30 |
| 2 | 100 | 537 | 0 | 0% | 8,096 | 8,900 | 13,000 | 9.04 |
| 2 | 200 | 664 | 515 | 77.56% | 13,574 | 15,000 | 15,000 | 11.38 |
| 3 | 50 | 608 | 0 | 0% | 3,145 | 3,700 | 5,100 | 10.25 |
| 3 | 100 | 533 | 1 | 0.19% | 8,272 | 10,000 | 13,000 | 8.98 |
| 3 | 200 | 704 | 463 | 65.77% | 12,645 | 15,000 | 15,000 | 11.88 |

At 50 users, three replicas improved average latency and attempted throughput over one replica, with no failures in either run. At 100 users, two replicas had no failures, compared with 41.25% for one replica. At 200 users, all configurations reached the client timeout frequently; three replicas had the lowest failure rate, but the local host was saturated. Attempts per second include requests that timed out, so throughput should be read together with the failure rate. These measurements are from local Docker, not Render production traffic.

## Demonstration video

The video shows the prediction flow, dataset visualizations, model status, upload/retraining process, and recorded Locust results. The retraining demonstration uses the local Docker deployment because Render’s free filesystem does not preserve uploaded data or retrained model changes across restarts.

## Repository structure

```text
notebook/flower_mlops.ipynb   Colab notebook for EDA, preprocessing, training, evaluation, and retraining
src/preprocessing.py          Dataset acquisition, splitting, image preprocessing, and upload validation
src/model.py                  Model building, training, evaluation, versioning, and retraining
src/prediction.py             Single-image prediction
src/database.py               SQLite upload records
src/insights.py               Image feature summaries for the dashboard
api/main.py                   FastAPI endpoints and background training jobs
ui/app.py                     Streamlit dashboard
models/                       Saved model, class names, and model metadata
Dockerfile.api                API container definition
Dockerfile.ui                 Dashboard container definition
docker-compose.yml            API replicas, Nginx load balancer, and dashboard
render.yaml                   Render API and dashboard services
locustfile.py                 Prediction/status load-test workload
scripts/                      Dataset download and load-test helpers
results/benchmark_summary.md  Full Locust measurements
```
