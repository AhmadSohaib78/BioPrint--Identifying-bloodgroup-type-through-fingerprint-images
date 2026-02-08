# backend/inference.py
import io
import os
import uuid
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
_MODEL = None

# We will apply Resize -> ToTensor -> manual Normalize([0.5],[0.5]) to match your test_transform.
to_tensor = transforms.ToTensor()
resize = transforms.Resize((224, 224))

def preprocess_to_tensor_and_vis(image_pil: Image.Image):
    """
    EXACT notebook preprocessing using OpenCV:
    - convert to grayscale
    - resize with INTER_AREA
    - CLAHE
    - GaussianBlur
    - Sharpen filter
    Returns:
      - tensor for model (1,3,224,224)
      - PIL image of enhanced fingerprint (uint8, 224x224)
    """

    import cv2
    import numpy as np
    from torchvision import transforms

    # Convert PIL → grayscale numpy
    img = np.array(image_pil.convert("L"))

    # === 1. Resize FIRST (INTER_AREA) ===
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # === 2. CLAHE ===
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # === 3. Gaussian blur ===
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # === 4. Sharpen ===
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharp)

    # === PREPROCESSED IMAGE (return this for preview) ===
    vis_pil = Image.fromarray(enhanced).convert("RGB")

    # === MODEL INPUT ===
    # Convert to tensor (still grayscale 1-channel)
    t = transforms.ToTensor()(enhanced)  # shape (1,224,224), values in [0,1]

    # Normalize EXACTLY like notebook
    t = (t - 0.5) / 0.5

    # Expand to 3 channels
    t_model = t.repeat(3, 1, 1)  # (3,224,224)
    input_tensor = t_model.unsqueeze(0).to(DEVICE)  # (1,3,224,224)

    return input_tensor, vis_pil


def load_model(weights_path="model/efficientnet_b0.pth", num_classes=8):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = models.efficientnet_b0(weights=None)
    # set classifier head like your notebook
    try:
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    except Exception:
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, num_classes))

    state = torch.load(weights_path, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    _MODEL = model
    return _MODEL

def predict_and_save_images(image_bytes: bytes, media_dir="media", weights_path="model/efficientnet_b0.pth"):
    """
    Saves original and preprocessed visual images to media_dir, runs model inference,
    returns filenames and model confidence.
    """
    if not os.path.exists(media_dir):
        os.makedirs(media_dir, exist_ok=True)

    orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    uid = uuid.uuid4().hex[:12]
    orig_name = f"original_{uid}.png"
    orig_path = os.path.join(media_dir, orig_name)
    orig_img.save(orig_path, format="PNG")

    # Preprocess according to your test_transform and create a visualization
    tensor_input, vis_pil = preprocess_to_tensor_and_vis(orig_img)

    pre_name = f"preprocessed_{uid}.png"
    pre_path = os.path.join(media_dir, pre_name)
    vis_pil.save(pre_path, format="PNG")

    model = load_model(weights_path=weights_path, num_classes=len(CLASS_NAMES))
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        model_conf = float(probs[idx])
        label = CLASS_NAMES[idx]

    return {
        "label": label,
        "model_confidence": model_conf,
        "original_filename": orig_name,
        "preprocessed_filename": pre_name
    }
