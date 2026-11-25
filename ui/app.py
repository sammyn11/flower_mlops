"""Streamlit dashboard: Predict | Visualizations | Uptime & Model | Upload & Retrain."""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Flower Classifier MLOps", page_icon="🌼", layout="wide")


def get(path, **kw):
    r = requests.get(f"{API}{path}", timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json()


page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 Visualizations", "🟢 Uptime & Model", "⬆️ Upload & Retrain"])
st.sidebar.caption(f"API: {API}")

# ------------------------------------------------------------------ Predict
if page == "🔮 Predict":
    st.title("🔮 Predict a flower")
    f = st.file_uploader("Upload one image (jpg/png)", type=["jpg", "jpeg", "png"])
    if f:
        c1, c2 = st.columns(2)
        c1.image(f, use_container_width=True)
        if c2.button("Predict", type="primary"):
            try:
                r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue(), f.type)}, timeout=60)
                if r.ok:
                    out = r.json()
                    c2.success(f"**{out['label']}**  ({out['confidence']:.1%} confidence)")
                    c2.caption(f"model v{out['model_version']} · {out['latency_ms']} ms · served by {out['served_by']}")
                    c2.bar_chart(pd.Series(out["probabilities"], name="probability"))
                else:
                    c2.error(r.json().get("detail", r.text))
            except Exception as e:
                c2.error(f"API unreachable: {e}")

# ------------------------------------------------------------------ Visualizations
elif page == "📊 Visualizations":
    st.title("📊 Dataset insights")
    try:
        d = get("/insights", timeout=120)
    except Exception as e:
        st.error(f"Could not load insights: {e}"); st.stop()

    # 1 - class distribution
    st.subheader("1. Class distribution")
    counts = pd.DataFrame(d["class_counts"]).fillna(0).astype(int)
    st.bar_chart(counts)
    tr = counts["train"]
    st.info(f"**Story:** `{tr.idxmax()}` is the largest class ({tr.max()} images) and `{tr.idxmin()}` the smallest "
            f"({tr.min()}), a {tr.max() / max(tr.min(), 1):.2f}× gap. The imbalance is mild, but a model can "
            f"drift toward the big class, so we report per-class recall/F1 and not just accuracy. "
            f"Uploaded retraining data appears as its own series.")

    # 2 - brightness
    st.subheader("2. Brightness per class")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    classes = list(d["brightness"].keys())
    ax.boxplot([d["brightness"][c] for c in classes], labels=classes)
    ax.set_ylabel("mean pixel intensity (0-255)")
    st.pyplot(fig)
    med = {c: float(np.median(v)) for c, v in d["brightness"].items() if v}
    hi, lo = max(med, key=med.get), min(med, key=med.get)
    st.info(f"**Story:** `{hi}` photos are the brightest (median {med[hi]:.0f}) and `{lo}` the darkest "
            f"({med[lo]:.0f}). Wide boxes mean inconsistent lighting inside a class, which is why we use "
            f"contrast/zoom augmentation. Brightness alone cannot separate classes, so the model must rely on shape and colour.")

    # 3 - colour channels
    st.subheader("3. Average colour profile per class")
    rgb = pd.DataFrame(d["rgb_mean"], index=["Red", "Green", "Blue"]).T
    st.bar_chart(rgb)
    share = rgb["Red"] / rgb.sum(axis=1)
    st.info(f"**Story:** every class is green-heavy because of foliage backgrounds, but `{share.idxmax()}` has the "
            f"highest red share ({share.max():.1%}) and `{share.idxmin()}` the lowest ({share.min():.1%}). "
            f"Colour is a useful cue (e.g. red/yellow petals), but overlapping profiles explain "
            f"why confusions happen between visually similar classes.")

    # 4 - image size
    st.subheader("4. Original image dimensions")
    rows = [(c, w, h, w / h) for c, s in d["size"].items() for w, h in s]
    sz = pd.DataFrame(rows, columns=["class", "width", "height", "aspect"])
    st.dataframe(sz.groupby("class")[["width", "height", "aspect"]].mean().round(2))
    st.info(f"**Story:** images average {sz.width.mean():.0f}×{sz.height.mean():.0f}px with mean aspect ratio "
            f"{sz.aspect.mean():.2f}. They are not square, so resizing to 224×224 stretches them slightly. "
            f"Acceptable here, but it motivates centre-crop/padding as a future improvement.")

# ------------------------------------------------------------------ Uptime & Model
elif page == "🟢 Uptime & Model":
    st.title("🟢 Uptime & model")
    if "pings" not in st.session_state:
        st.session_state.pings = []
    st.button("Refresh")
    try:
        t0 = time.perf_counter()
        s = get("/status")
        lat = (time.perf_counter() - t0) * 1000
        st.session_state.pings.append({"time": time.strftime("%H:%M:%S"), "up": 1, "latency_ms": lat})
        up = s["uptime_seconds"]
        c = st.columns(4)
        c[0].metric("API", "UP ✅")
        c[1].metric("Uptime", f"{int(up // 3600)}h {int(up % 3600 // 60)}m {int(up % 60)}s")
        c[2].metric("Model version", s["model_version"])
        c[3].metric("Ping latency", f"{lat:.0f} ms")
        st.caption(f"Served by container: {s['served_by']}")
        if s["model_metrics"]:
            st.subheader("Latest test-set metrics")
            m = s["model_metrics"]
            c = st.columns(5)
            for col, k in zip(c, ["accuracy", "loss", "precision", "recall", "f1"]):
                col.metric(k, f"{m[k]:.4f}")
        st.subheader("Retraining triggers")
        st.write(f"Pending uploads: **{s['pending_uploads']} / {s['retrain_threshold']}** (auto-retrain at threshold)")
        if s["avg_recent_confidence"] is not None:
            st.write(f"Mean confidence of recent predictions: **{s['avg_recent_confidence']:.2%}**")
        if s["retrain_recommended"]:
            st.warning("⚠️ Confidence is low - model drift suspected. Consider retraining.")
    except Exception as e:
        st.session_state.pings.append({"time": time.strftime("%H:%M:%S"), "up": 0, "latency_ms": None})
        st.error(f"API is DOWN: {e}")
    if st.session_state.pings:
        st.subheader("Ping history")
        st.line_chart(pd.DataFrame(st.session_state.pings).set_index("time")["latency_ms"])

# ------------------------------------------------------------------ Upload & Retrain
else:
    st.title("⬆️ Upload data & retrain")
    try:
        s = get("/status")
        classes = (get("/metrics") or {}).get("class_names", [])
    except Exception as e:
        st.error(f"API unreachable: {e}"); st.stop()

    st.subheader("1. Upload labelled images (bulk)")
    label = st.selectbox("Class label for this batch", classes)
    files = st.file_uploader("Select multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files and st.button(f"Upload {len(files)} images"):
        with st.spinner("Validating, resizing and saving..."):
            r = requests.post(f"{API}/upload", data={"label": label},
                              files=[("files", (f.name, f.getvalue(), f.type)) for f in files], timeout=600)
        if r.ok:
            o = r.json()
            st.success(f"Saved & preprocessed {o['saved']} images (rejected {len(o['rejected'])}). "
                       f"Pending: {o['pending_uploads']}/{o['threshold']}")
            if o["auto_retrain_triggered"]:
                st.info("Threshold reached -> automatic retraining started.")
        else:
            st.error(r.json().get("detail", r.text))

    st.subheader("2. Train / retrain")
    st.write(f"Pending uploads: **{s['pending_uploads']}** · total uploads: **{s['total_uploads']}**")
    c1, c2 = st.columns(2)
    go_retrain = c1.button("🔁 Retrain on uploaded data", type="primary")
    go_train = c2.button("🆕 Train from scratch (ImageNet weights)")
    if go_retrain or go_train:
        r = requests.post(f"{API}/{'retrain' if go_retrain else 'train'}", timeout=30)
        if not r.ok:
            st.error(r.json().get("detail", r.text))

    job = get("/job")
    if job["status"] == "running":
        box = st.empty()
        while True:
            job = get("/job")
            box.code(f"STATUS: {job['status']} | STAGE: {job['stage']}\n" + "\n".join(job["logs"][-15:]))
            if job["status"] != "running":
                break
            time.sleep(3)
    elif job["logs"]:
        st.code(f"STATUS: {job['status']}\n" + "\n".join(job["logs"][-15:]))
    if job["status"] == "done":
        st.success("New model saved and live.")
    elif job["status"] == "error":
        st.error(job["error"])
