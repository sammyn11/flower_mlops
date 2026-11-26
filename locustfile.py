"""Load test:  locust -f locustfile.py --host http://localhost:8080"""
import glob
import random
from locust import HttpUser, between, task

IMAGES = sorted(glob.glob("data/test/**/*.jpg", recursive=True))[:200]
if not IMAGES:
    raise RuntimeError("No test images found under data/test/**/*.jpg")
# Read a small image sample once. Reading files separately in each user's
# on_start makes Windows bind mounts spend most of a short test spawning users.
SAMPLE_IMAGES = [open(p, "rb").read() for p in random.sample(IMAGES, min(10, len(IMAGES)))]


class FlowerUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        self.imgs = SAMPLE_IMAGES

    @task(9)
    def predict(self):
        self.client.post("/predict", files={"file": ("flower.jpg", random.choice(self.imgs), "image/jpeg")},
                         name="/predict", timeout=15)

    @task(1)
    def status(self):
        self.client.get("/status", name="/status", timeout=15)
