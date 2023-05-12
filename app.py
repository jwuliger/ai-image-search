import os
import torch
import clip
import pickle
import base64
from PIL import Image
import streamlit as st
from io import BytesIO
from typing import List, Tuple

# Load the pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

root_directory = 'C:/Users/jared/Pictures/MidJourney'
features_file = 'image_features.pkl'
images_per_page = 5


@st.cache_data
def image_to_base64(img_path: str) -> str:
    """Convert an image to a base64 string."""
    with Image.open(img_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_features(img_path: str) -> torch.Tensor:
    """Extract features from an image using the CLIP model."""
    img = Image.open(img_path).convert("RGB")
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(img_preprocessed)


def load_or_extract_features() -> dict:
    """Load features from a pickle file, or extract them if the file doesn't exist."""
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            return pickle.load(f)

    st.write("Image features file not found. Extracting features, please wait...")
    image_features = {
        os.path.join(dirpath, filename): extract_features(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(root_directory)
        for filename in filenames
        if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    }
    with open(features_file, 'wb') as f:
        pickle.dump(image_features, f)

    st.write("Image features extraction completed.")
    return image_features


def search_images(keyword: str, image_features: dict, threshold: int = 28) -> List[Tuple[float, str]]:
    """Search for images based on a keyword."""
    text = clip.tokenize(keyword).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    return sorted(
        ((similarity, img_path) for img_path, img_features in image_features.items()
         if (similarity := (img_features @ text_features.T).item()) >= threshold),
        key=lambda x: x[0],
        reverse=True
    )


def run_app(image_features: dict):
    """Streamlit app to search for images based on a keyword."""
    st.title("JW's AI Image Search")
    if keyword := st.text_input("Enter the keyword to search for:"):
        matching_images = search_images(keyword, image_features)
        st.write(f"{len(matching_images)} images found:")

        num_pages = len(matching_images) // images_per_page + \
            bool(len(matching_images) % images_per_page)

        st.session_state.page_number = st.session_state.get('page_number', 1)

        col1, col2 = st.columns(2)

        if col1.button('Previous') and st.session_state.page_number > 1:
            st.session_state.page_number -= 1

        if col2.button('Next') and st.session_state.page_number < num_pages:
            st.session_state.page_number += 1
            st.write(
                f"You are on page {st.session_state.page_number} of {num_pages}.")

        start = (st.session_state.page_number - 1) * images_per_page
        end = start + images_per_page

        for similarity, img in matching_images[start:end]:
            img_base64 = image_to_base64(img)
            st.markdown(
                f'<img src="data:image/png;base64,{img_base64}" style="width: 100%">', unsafe_allow_html=True)
            st.write(f"Similarity: {similarity:.4f}")
            st.text_input('File path:', value=img)


if __name__ == "__main__":
    image_features = load_or_extract_features()
    run_app(image_features)
