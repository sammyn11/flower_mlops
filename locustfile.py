"""Load test:  locust -f locustfile.py --host http://localhost:8080"""
import glob
import random
from locust import HttpUser, between, task

IMAGES = sorted(glob.glob("data/test/**/*.jpg", recursive=True))[:200]


class FlowerUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        self.imgs = [open(p, "rb").read() for p in random.sample(IMAGES, min(10, len(IMAGES)))]

    @task(9)
    def predict(self):
        self.client.post("/predict", files={"file": ("flower.jpg", random.choice(self.imgs), "image/jpeg")},
                         name="/predict")

    @task(1)
    def status(self):
        self.client.get("/status", name="/status")
