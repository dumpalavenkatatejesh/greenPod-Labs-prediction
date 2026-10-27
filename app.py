from pathlib import Path
import tempfile

import joblib
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from train_and_evaluate import CLASSES, SmallCNN, classical_features

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
SVM_PATH = OUTPUT_DIR / "svm_hog_hsv.joblib"
CNN_PATH = OUTPUT_DIR / "small_cnn_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]

st.set_page_config(page_title="Orange Condition Classifier", page_icon="🍊", layout="wide")

@st.cache_resource
def load_svm():
    if not SVM_PATH.exists():
        raise FileNotFoundError(f"SVM model not found: {SVM_PATH}")
    return joblib.load(SVM_PATH)

@st.cache_resource
def load_cnn():
    if not CNN_PATH.exists():
        raise FileNotFoundError(f"CNN weights not found: {CNN_PATH}. Run training first.")
    model = SmallCNN(len(CLASSES)).to(DEVICE)
    state = torch.load(CNN_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_svm(image: Image.Image):
    # Reuse the exact classical_features() implementation used during training.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        image.convert("RGB").save(temp_path, format="PNG")
        features = classical_features(str(temp_path)).reshape(1, -1)
    finally:
        temp_path.unlink(missing_ok=True)
    model = load_svm()
    label = str(model.predict(features)[0])
    probs = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
    names = [str(x) for x in model.classes_] if probs is not None else []
    return label, probs, names

def predict_cnn(image: Image.Image):
    model = load_cnn()
    tensor = CNN_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs, CLASSES

st.title("Fruits Classifier")
st.caption("Compare a classical SVM (HSV + HOG features) with a small CNN (learned image features).")
st.warning("Dataset note: this project classifies orange image categories (blackspot, canker, fresh, grenning), not avocado ripening stages. Predictions are estimates, not a food-safety assessment.")

with st.sidebar:
    st.header("Models")
    st.write("**SVM:** HSV colour histogram + HOG features")
    st.write("**CNN:** four convolution blocks, trained on 64 × 64 images")
    st.caption(f"Inference device: {DEVICE.type.upper()}")
    st.markdown("**Saved model files required**")
    st.code("outputs/svm_hog_hsv.joblib\noutputs/small_cnn_best.pt")

uploaded = st.file_uploader("Upload an orange image", type=IMAGE_EXTENSIONS)
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    col1, col2 = st.columns(2)
    for col, model_name, predictor in [
        (col1, "SVM · HSV + HOG", predict_svm),
        (col2, "Small CNN", predict_cnn),
    ]:
        with col:
            st.subheader(model_name)
            try:
                label, probs, names = predictor(image)
                st.success(f"Predicted class: {label.upper()}")
                if probs is not None:
                    confidence = float(probs[list(names).index(label)]) if model_name.startswith("SVM") else float(probs[CLASSES.index(label)])
                    st.metric("Estimated confidence", f"{confidence:.1%}")
                    st.caption("Model probability estimate; not a guarantee of correctness.")
                    st.write("Class probability estimates")
                    for name, prob in sorted(zip(names, probs), key=lambda item: item[1], reverse=True):
                        st.write(f"{name}: {float(prob):.1%}")
                        st.progress(float(prob))
                elif model_name.startswith("SVM"):
                    st.warning("The saved SVM needs retraining to show probability estimates.")
            except Exception as exc:
                st.error(f"Could not run {model_name}: {exc}")
else:
    st.info("Upload a JPG, PNG, BMP, or WebP image to see predictions.")

st.divider()
st.subheader("Held-out test-set comparison")
st.caption("Previously measured results; these metrics are not recalculated from the uploaded image.")
st.dataframe([
    {"Metric": "Accuracy", "SVM (HSV + HOG)": "90.91%", "Small CNN": "90.91%"},
    {"Metric": "Macro precision", "SVM (HSV + HOG)": "90.30%", "Small CNN": "90.62%"},
    {"Metric": "Macro recall", "SVM (HSV + HOG)": "90.15%", "Small CNN": "90.91%"},
    {"Metric": "Macro F1-score", "SVM (HSV + HOG)": "90.03%", "Small CNN": "90.40%"},
    {"Metric": "Weighted F1-score", "SVM (HSV + HOG)": "90.80%", "Small CNN": "90.77%"},
], hide_index=True, use_container_width=True)
st.caption("The test set contains 99 images, so results on new images may differ. The required assignment dataset was avocado; this implementation currently uses the orange dataset present in this project.")
