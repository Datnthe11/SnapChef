import sys
import os
import urllib.request
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import streamlit as st
import time

# Import project modules
from model.args import get_parser
from model.model import get_model
from model.output_utils import prepare_output

# Page Configuration
st.set_page_config(
    page_title="SnapChef - AI Recipe Discovery",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- CUSTOM CSS FOR PREMIUM LOOK (ENGLISH) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    /* Global Typography */
    .main, .stApp, .stMarkdown, p, div {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Enforce Dark Theme Background */
    .stApp {
        background-color: #0f172a !important;
        background-image: radial-gradient(circle at 2% 2%, #1e293b 0%, #0f172a 100%) !important;
        color: #f8fafc !important;
    }

    /* Fixed Header Styles */
    h1 { font-size: 3.5rem !important; font-weight: 800 !important; color: #ffffff !important; }
    h2 { font-size: 2.2rem !important; color: #ffffff !important; }
    h3 { font-size: 1.8rem !important; color: #ffffff !important; margin-bottom: 1rem !important; }
    h4 { font-size: 1.4rem !important; color: #fbbf24 !important; font-weight: 700 !important; }

    /* Custom Ingredient Tags */
    .ingredient-tag {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 10px;
        margin: 4px;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Custom Step Boxes */
    .step-box {
        background: #1e293b;
        padding: 24px;
        border-left: 4px solid #fbbf24;
        border-radius: 12px;
        margin-bottom: 16px;
        line-height: 1.6;
        color: #f1f5f9;
        font-size: 1.15rem;
    }

    .step-number {
        color: #fbbf24;
        font-weight: 800;
        font-size: 1rem;
        text-transform: uppercase;
        display: block;
        margin-bottom: 4px;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: #10b981 !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        transition: opacity 0.3s;
    }

    .stButton>button:hover {
        opacity: 0.9;
    }

    /* Center Titles */
    .title-container {
        text-align: center;
        margin-bottom: 40px;
    }

    /* Fix image sizing */
    .stImage > img {
        border-radius: 16px;
        max-height: 400px !important;
        width: auto !important;
        margin: 0 auto;
        display: block;
    }

    /* Fix the Info Box (Tip) readability */
    .stAlert {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    .stAlert div {
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- CORE LOGIC ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_file(url, save_path):
    with st.spinner(f"⏳ Downloading system file: {os.path.basename(save_path)}..."):
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            st.error(f"❌ Download error: {e}")
            raise e

@st.cache_resource()
def load_resources():
    parser = get_parser()
    args = parser.parse_args([])
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    model_path = os.path.join(data_dir, "modelbest.ckpt")
    ingr_vocab_path = os.path.join(data_dir, "ingr_vocab.pkl")
    instr_vocab_path = os.path.join(data_dir, "instr_vocab.pkl")
    
    urls = {
        model_path: "https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt",
        ingr_vocab_path: "https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl",
        instr_vocab_path: "https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl"
    }

    for path, url in urls.items():
        if not os.path.exists(path):
            download_file(url, path)
    
    with open(ingr_vocab_path, "rb") as f:
        ingr_vocab = pickle.load(f)

    with open(instr_vocab_path, "rb") as f:
        instr_vocab = pickle.load(f)
    
    if isinstance(ingr_vocab, list):
        ingr_vocab_list = ingr_vocab
        ingr_vocab_dict = {i: word for i, word in enumerate(ingr_vocab_list)}
    else:
        ingr_vocab_list = list(ingr_vocab.values())
        ingr_vocab_dict = ingr_vocab

    # Load Model
    model = get_model(args, len(ingr_vocab_dict), len(instr_vocab))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, ingr_vocab_list, instr_vocab

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# --- UI RENDER ---

logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
with logo_col2:
    st.image("assets/logo.png", width=150)
    st.markdown("""
        <div class="title-container" style="padding-top: 0;">
            <h1>SnapChef</h1>
            <p>AI-Powered Recipe Reconstruction from Food Images</p>
        </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.subheader("📸 Upload Food Image")
    # Fix the empty label warning
    uploaded_file = st.file_uploader("Upload your dish photo", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width='stretch', caption="Your selected dish")
        predict_btn = st.button("🌟 Start Analysis")
    else:
        st.info("💡 Tip: Use a high-quality photo of a single dish for the most accurate results.")
        predict_btn = False

with col2:
    if predict_btn and uploaded_file:
        with st.status("🧠 AI is analyzing the ingredients and instructions...", expanded=True) as status:
            try:
                model, ingr_vocab, instr_vocab = load_resources()
                st.write("✨ Model ready.")
                
                input_tensor = preprocess_image(image)
                st.write("🔍 Extracting visual cues...")
                
                with torch.no_grad():
                    output = model.sample(input_tensor)
                    
                if not output or "recipe_ids" not in output or len(output["recipe_ids"]) == 0:
                    st.error("Sorry, we couldn't generate a recipe for this image.")
                else:
                    ingr_ids = output['ingr_ids'].cpu().numpy()
                    recipe_ids = output['recipe_ids'].cpu().numpy()
                    
                    recipe, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, instr_vocab)
                    
                    if not valid['is_valid']:
                        st.warning(f"Note: Result summary: {valid['reason']}")
                    
                    status.update(label="✅ Recipe Generated!", state="complete", expanded=False)
                    
                    # Display Results
                    st.markdown(f"### 🍽️ {recipe.get('title', 'SUGGESTED RECIPE').upper()}")
                    
                    st.markdown("#### 🥕 Predicted Ingredients")
                    ingr_html = "".join([f'<span class="ingredient-tag">{ing}</span>' for ing in recipe["ingrs"]])
                    st.markdown(f'<div style="margin-bottom: 2rem;">{ingr_html}</div>', unsafe_allow_html=True)
                    
                    st.markdown("#### 📜 Cooking Instructions")
                    for i, step in enumerate(recipe["recipe"]):
                        st.markdown(f"""
                            <div class="step-box">
                                <span class="step-number">Step {i+1}</span><br>{step}
                            </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
                status.update(label="❌ Error", state="error")
    else:
        st.markdown('<div style="text-align: center; color: #64748b; margin-top: 5rem; opacity: 0.6;">'
                    '<h3>Waiting for Input</h3>'
                    '<p>Upload an image to reveal the recipe</p>'
                    '</div>', unsafe_allow_html=True)
