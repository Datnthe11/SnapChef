import sys
import os
import urllib.request
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from model.args import get_parser
from model.model import get_model
from model.output_utils import prepare_output

# Determine device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to download file from URL
def download_file(url, save_path):
    st.info(f"⏳ Downloading file from {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        st.success(f"✅ Download complete: {os.path.basename(save_path)}")
    except Exception as e:
        st.error(f"❌ Download error: {e}")
        raise e

# Function to load model
def load_model(model_path, args, ingr_vocab_size, instr_vocab_size, device):
    model = get_model(args, ingr_vocab_size, instr_vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function for image preprocessing
def preprocess_image(image, model_name="resnet101"):
    image_size = 299 if "inception" in model_name else 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Function to predict recipe
def predict_recipe(image, model, ingr_vocab, instr_vocab):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model.sample(input_tensor)
        if not output or "recipe_ids" not in output or len(output["recipe_ids"]) == 0:
            raise ValueError("Model failed to generate a recipe. Please check input and model.")
        
        ingr_ids = output['ingr_ids'].cpu().numpy()
        recipe_ids = output['recipe_ids'].cpu().numpy()
        
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, instr_vocab)
        
        if not valid['is_valid']:
            raise ValueError(f"Invalid recipe generated. Reason: {valid['reason']}")
        
        return outs

# Load model and vocabularies
@st.cache_resource()
def load_resources():
    parser = get_parser()
    args = parser.parse_args([])  # Avoid errors when running on Streamlit
    
    # File paths (using relative paths)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    model_path = os.path.join(data_dir, "modelbest.ckpt")
    ingr_vocab_path = os.path.join(data_dir, "ingr_vocab.pkl")
    instr_vocab_path = os.path.join(data_dir, "instr_vocab.pkl")
    
    # Weight file download URLs
    urls = {
        model_path: "https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt",
        ingr_vocab_path: "https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl",
        instr_vocab_path: "https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl"
    }

    # Check and download missing files
    for path, url in urls.items():
        if not os.path.exists(path):
            st.warning(f"File {os.path.basename(path)} not found. Starting download...")
            download_file(url, path)
    
    # Load vocabs
    with open(ingr_vocab_path, "rb") as f:
        ingr_vocab = pickle.load(f)

    with open(instr_vocab_path, "rb") as f:
        instr_vocab = pickle.load(f)
    
    if isinstance(ingr_vocab, list):
        ingr_vocab_list = ingr_vocab
        ingr_vocab = {i: word for i, word in enumerate(ingr_vocab_list)}
    else:
        ingr_vocab_list = list(ingr_vocab.values())

    model = load_model(model_path, args, len(ingr_vocab), len(instr_vocab), device)
    
    return model, ingr_vocab_list, instr_vocab

# Load resources
model, ingr_vocab, instr_vocab = load_resources()

# Streamlit UI
st.title("🍽️ Inverse Cooking - Recipe Prediction from Images")
st.write("Upload a dish photo to get a suggested recipe!")

# Upload image
uploaded_file = st.file_uploader("📸 Choose a food image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Dish", width='stretch')
    
    if st.button("🎯 Predict Recipe"):
        st.write("⏳ Running model analysis...")
        
        try:
            recipe = predict_recipe(image, model, ingr_vocab, instr_vocab)
            st.write("✅ **Analysis Complete!**")

            st.subheader("🥕 Ingredients:")
            for ingredient in recipe["ingrs"]:
                st.write(f"- {ingredient}")

            st.subheader("📜 Cooking Instructions:")
            for step in recipe["recipe"]:
                st.write(f"- {step}")

        except Exception as e:
            st.error(f"Error: {e}")
