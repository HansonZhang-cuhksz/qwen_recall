import torch
import open_clip
from PIL import Image
import os
import streamlit as st

@st.cache_resource
def init(force_cpu=False):
    print("ViT model loading...")
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    # Load CLIP model and preprocessing
    os.environ["TORCH_HOME"] = os.path.dirname(__file__)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=r".\clipModels\open_clip_model.safetensors")
    model = model.to(device).eval()
    print("ViT model loaded successfully.")
    return model, preprocess, device

def get_clip_features(img_path, model_preprocess_device):
    model, preprocess, device = model_preprocess_device

    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # Normalize
    return features

def get_similarity(img1_path, img2_path, image_diff_driver):
    feat1 = get_clip_features(img1_path, image_diff_driver)
    feat2 = get_clip_features(img2_path, image_diff_driver)
    similarity = (feat1 @ feat2.T).item()  # Cosine similarity
    return similarity