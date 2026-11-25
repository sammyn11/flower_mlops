# Flower Classification MLOps Project

## About my project

I built an end-to-end image classification pipeline for five flower classes: daisy, dandelion, rose, sunflower, and tulip. The project uses the TF Flowers image dataset and a MobileNetV2 model pre-trained on ImageNet. It includes preprocessing, training, evaluation, a prediction API, a web dashboard, bulk image uploads, retraining, monitoring, Docker deployment, and Locust load testing.

## Project links

- GitHub repository: **I will add my repository URL after publishing it.**
- Video demonstration: **I will add my YouTube link after recording the demo.**
- Deployed dashboard: **I will add my public URL after deployment.**
- API documentation: **I will add my public `/docs` URL after deployment.**

I will replace these placeholders with the real links before submitting. I will also add the real model metrics and load-test results after running those steps; I will not report sample or estimated results as measured results.

## What the project does

- Predicts the flower class and confidence from one uploaded image.
- Shows class counts, image brightness, average RGB colour, and image dimensions, with an interpretation of each feature.
- Accepts multiple labeled images for retraining, validates and resizes them, saves them to disk, and records them in SQLite.
- Retrains the existing model using uploaded images plus a replay sample from the original training data.
- Shows API uptime, model version and evaluation metrics, pending uploads, recent prediction confidence, and training-job status.
- Provides a FastAPI endpoint and a Streamlit dashboard.
- Uses Docker Compose and Nginx to run and load-balance multiple API containers.
- Uses Locust to measure request latency and throughput at different user and container counts.

## Step 1: Put the project in GitHub

1. Create a new, empty repository in my GitHub account.
2. Upload or push all project files, including this README, the `src/`, `api/`, `ui/`, `notebook/`, `scripts/`, and Docker files.
3. Do not upload `.venv/`, `data/raw/`, `data/uploads/`, `data/uploads.db`, or cloud credentials.
4. Copy the GitHub repository URL into the Project links section above.

If I am using Git from this project folder, I can run:

```bash
git init
git add .
git commit -m "Build flower classification MLOps pipeline"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

I will replace `USERNAME` and `REPOSITORY` with my GitHub details. If Git reports that a remote already exists, I will check it with `git remote -v` instead of adding it again.

## Step 2: Train and evaluate the model in Google Colab

1. Open Google Colab and choose **Runtime → Change runtime type → T4 GPU** (or another available GPU).
2. Download my GitHub repository as a ZIP. In Colab, open the **Files** sidebar, upload the ZIP under `/content`, and extract it there. The extracted project folder must contain `src/`, `notebook/`, and `requirements-colab.txt`.
3. Open `notebook/flower_mlops.ipynb` in Colab. If Colab opened the notebook separately, keep the extracted project folder under `/content`; the notebook setup cell searches there for `src/config.py`.
4. If a dependency import fails, run this in a notebook cell from the project root:

   ```python
   %pip install -r requirements-colab.txt
   ```

   TensorFlow is normally preinstalled in Colab, so `requirements-colab.txt` intentionally does not reinstall it.

5. Run **Runtime → Run all**. The notebook downloads TF Flowers, makes a stratified train/test split, displays the data visualizations, preprocesses the images, trains a scratch CNN baseline and the MobileNetV2 model, and evaluates both on the held-out test set.
6. Save the executed notebook with its outputs. Record the actual accuracy, loss, macro precision, macro recall, macro F1, per-class report, and confusion matrix in the results section below. Use the confusion matrix and feature visualizations to write your own findings.
7. The notebook saves `flower_model.h5`, `class_names.json`, and `model_meta.json` in `models/`. Download those files from Colab or copy them to Google Drive. Put all three into this project's `models/` folder before starting the API or Docker deployment.
8. To demonstrate retraining inside the notebook, set `RUN_RETRAIN_DEMO = True` in the last section and run that cell. The actual app retraining demonstration is in Step 4.

The hyperparameter search and two training phases may take time. Use the GPU runtime and keep the notebook open until training and artifact saving finish. If the Colab session disconnects, reconnect and rerun the setup and training cells as needed.

## Step 3: Run the API and dashboard locally

I need Python 3.10 or 3.11 for the pinned TensorFlow 2.15 environment. I also need the three model artifacts from Step 2.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and split the dataset for local visualizations and retraining replay:

```bash
python scripts/download_data.py
```

Start the API in the first terminal:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start the dashboard in a second terminal. On PowerShell:

```powershell
$env:API_URL="http://localhost:8000"
streamlit run ui/app.py
```

On macOS/Linux:

```bash
API_URL=http://localhost:8000 streamlit run ui/app.py
```

Open the dashboard at `http://localhost:8501` and the interactive API docs at `http://localhost:8000/docs`.

## Step 4: Demonstrate prediction, upload, and retraining

1. Open the dashboard's **Predict** page.
2. Upload a flower JPG or PNG and click **Predict**. Show the predicted class, confidence, probability chart, model version, latency, and serving container.
3. Open **Visualizations**. Review the class distribution, brightness, RGB colour profile, and image dimensions. Explain at least three features and what they suggest about the dataset.
4. Open **Upload & Retrain** and select a flower class for which I have new labeled examples.
5. Upload several images that genuinely belong to that class and click **Upload**. Confirm the accepted/rejected counts and pending-upload count. The API validates each file, converts it to RGB, resizes it to 224×224, saves it under `data/uploads/<class>/`, and records it in SQLite.
6. Click **Retrain on uploaded data**. Show the job logs as the current saved model is loaded, the new data is combined with replay images, training runs, and a new model version is evaluated and saved.
7. After the job completes, make another prediction and show the new model version on the **Uptime & Model** page.

The API also starts retraining automatically when pending uploads reach `RETRAIN_THRESHOLD` (default: 50). The confidence monitor can recommend retraining when recent confidence is low; confidence alone is not proof of model drift because the app does not have ground-truth labels for every prediction.

## Step 5: Run with Docker and scale API containers

I need Docker Engine and the Docker Compose plugin. Before starting, I must have the model artifacts in `models/` and the dataset in `data/`.

Build and start two API containers, Nginx, and the dashboard:

```bash
docker compose up -d --build --scale api=2
```

- Dashboard: `http://localhost:8501`
- Load-balanced API docs: `http://localhost:8080/docs`
- API health/status: `http://localhost:8080/status`

The Compose configuration shares `data/` and `models/` between the API containers. To stop the services:

```bash
docker compose down
```

## Step 6: Deploy to a cloud VM

This project uses a single Linux VM as a straightforward cloud deployment target. I need an account with a provider such as Google Cloud or AWS; those accounts and resources are not created by this repository.

1. Create a Linux VM with enough memory and disk for TensorFlow, Docker, the dataset, and multiple model-serving containers.
2. Configure the VM firewall/security group to allow inbound TCP traffic on ports **8501** (dashboard) and **8080** (API/load balancer). Restrict access where practical.
3. Install Docker Engine and the Docker Compose plugin on the VM.
4. Clone my GitHub repository onto the VM.
5. Copy the three trained model artifacts into the VM's `models/` folder. Do not put cloud keys or passwords in the repository.
6. Download the dataset and start the containers:

   ```bash
   python3 scripts/download_data.py
   docker compose up -d --build --scale api=2
   ```

7. Open `http://PUBLIC_VM_IP:8501` for the dashboard and `http://PUBLIC_VM_IP:8080/docs` for the API docs. Replace `PUBLIC_VM_IP` with the VM's actual public IP or DNS name.
8. Add the working URLs to the Project links section. Keep the VM running while recording the demo and running the cloud load test.

This single-VM setup demonstrates deployment and container scaling for the assignment; it is not a highly available production architecture. Cloud hosting may incur charges, so I will stop or delete the VM when I have finished if I no longer need it.

## Step 7: Run the Locust flood-request simulation

Run the load test from a machine with Docker Compose and Locust installed, with the Docker deployment running and the test images downloaded. The provided script requires Bash, so on Windows I should use WSL with Docker access.

macOS/Linux/WSL:

```bash
python -m pip install locust
chmod +x scripts/run_load_tests.sh
./scripts/run_load_tests.sh
```

The script runs 1, 2, and 3 API containers with 50, 100, and 200 concurrent users. It saves Locust CSV files in `results/` and builds `results/summary.md`. I will include the measured request count, failures, average and median latency, p95 latency, and requests per second in my report. I will explain how latency and throughput change as I add containers and users. The results depend on the VM and network, so I must run the test and report its actual output.

If the script cannot start Docker or Locust, I will check that Docker is running, the API is reachable at `http://localhost:8080/status`, and the dataset images exist under `data/test/` before retrying.

## Step 8: Record my video demonstration

I will record a clear demonstration with my camera on and sound working. In the video I will:

1. Introduce the flower classification problem and briefly explain the dataset and model.
2. Show the notebook's preprocessing, training, and evaluation outputs.
3. Upload one image on the dashboard and show the prediction and confidence.
4. Explain at least three visualized image features and the story they tell.
5. Upload a batch of labeled images and show that they are saved.
6. Trigger retraining, show the job progress, and show the resulting model version/metrics.
7. Show the deployed dashboard and API, then show the Locust results for different container counts.
8. Upload the recording to YouTube and replace the video placeholder above with its link.

## Results to complete after running the project

### Notebook evaluation

Paste the actual held-out test metrics from my executed notebook here. Include both baseline and MobileNetV2 values and discuss the most confused classes using the confusion matrix.

| Model | Accuracy | Loss | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | Add measured value | Add measured value | Add measured value | Add measured value | Add measured value |
| MobileNetV2 | Add measured value | Add measured value | Add measured value | Add measured value | Add measured value |

### Locust results

Paste the measured rows from `results/summary.md` here and describe what changed as container count and user load increased.

| API containers | Concurrent users | Requests | Failures | Average (ms) | Median (ms) | p95 (ms) | Requests/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Add measured value | Add measured value | Add measured value | Add measured value | Add measured value | Add measured value | Add measured value | Add measured value |

## Submission checklist

- [ ] Publish the repository and add its GitHub URL.
- [ ] Run the Colab notebook and save it with outputs and real metric interpretations.
- [ ] Include the model artifacts or provide a reliable download location for them.
- [ ] Demonstrate single-image prediction, visualizations, bulk upload, and retraining.
- [ ] Deploy the app and add working dashboard/API URLs.
- [ ] Run Locust and include real results for different API container counts.
- [ ] Record and publish the camera-on video, then add the YouTube URL.
- [ ] Submit a ZIP of the repository for the first attempt and the GitHub repository URL for the second attempt.

## Repository structure

```text
notebook/flower_mlops.ipynb   Colab notebook for EDA, preprocessing, training, evaluation, and retraining
src/preprocessing.py          Dataset acquisition, splitting, image preprocessing, upload validation
src/model.py                  Model building, training, evaluation, versioning, and retraining
src/prediction.py             Single-image prediction
src/database.py               SQLite upload records
src/insights.py               Image feature summaries for the dashboard
api/main.py                   FastAPI endpoints and background training jobs
ui/app.py                     Streamlit dashboard
Dockerfile.api                API container definition
Dockerfile.ui                 Dashboard container definition
docker-compose.yml             API replicas, Nginx load balancer, and dashboard
locustfile.py                 Prediction/status load-test workload
scripts/                      Dataset download and load-test helpers
```
