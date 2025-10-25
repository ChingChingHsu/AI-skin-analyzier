import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# --- 導入必要的函式庫 ---
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# --- 核心功能定義 ---
@st.cache_resource
def load_model():
    """
    載入訓練好的 PyTorch 模型並設定為評估模式。
    """
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 7),
        nn.LogSoftmax(dim=1)
    )
    try:
        model.load_state_dict(torch.load('best_skin_cancer_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Error: Model weights file 'best_skin_cancer_model.pth' not found!")
        st.error("Please ensure the trained model weights file is in the same folder as this app.py.")
        return None
    model.eval()
    return model

def process_image(image_pil):
    """
    對 PIL 影像進行預處理，返回 PyTorch 張量。
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT
    transform = weights.transforms()
    image_tensor = transform(image_pil)
    return image_tensor

def predict(image_tensor, model, topk=5):
    """
    使用模型進行預測，返回機率和類別索引。
    """
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model.forward(image_tensor)
    ps = torch.exp(output)
    top_p, top_class_idx = ps.topk(topk, dim=1)
    return top_p[0].numpy(), top_class_idx[0].numpy()

# --- Streamlit 網頁介面 (UI) ---

idx_to_lesion_name = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}
class_names = list(idx_to_lesion_name.values())

# 載入模型
model = load_model()

# 網頁標題和說明文字設定
st.title("AI-Assisted Skin Lesion Identification System")
st.write("Please upload an image of a skin lesion, and the AI model will predict its possible type.")
st.caption("Note: The system prediction results are for reference only. Please consult your attending physician for an accurate diagnosis.")

# 設定信心度閾值
CONFIDENCE_THRESHOLD = 0.7
st.sidebar.info(f"If the highest prediction probability is below {CONFIDENCE_THRESHOLD*100:.0f}%, it will be classified as 'Unable to identify'.")

# 建立一個圖片上傳器
uploaded_file = st.file_uploader("Upload an image for AI-assisted identification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # 設定圖片標題
    st.image(image, caption='Uploaded Image')
    st.write("")

    # 設定按鈕文字
    if st.button('Start Prediction', use_container_width=True):
        # 設定等待訊息
        with st.spinner('Model is analyzing, please wait...'):
            image_tensor = process_image(image)
            probs, classes_idx = predict(image_tensor, model)

            highest_prob = probs[0]

            if highest_prob < CONFIDENCE_THRESHOLD:
                # 設定警告訊息
                st.warning(f"Unable to identify as a known skin lesion (Highest Confidence: {highest_prob*100:.1f}%)", icon="⚠️")
                st.write("Please ensure the uploaded image is clear and is indeed a skin lesion image.")
            else:
                class_labels = [class_names[i] for i in classes_idx]
                pred_label_name = class_labels[0]

                # 設定預測結果標題和信心度標籤
                st.subheader(f"AI Prediction Result: **{pred_label_name}** (Confidence: {highest_prob*100:.1f}%)")
                # 設定圖表說明文字
                st.write("Top 5 Prediction Probability Distribution:")
                fig, ax = plt.subplots()
                # 設定 class_labels 作為 Y 軸標籤
                sns.barplot(x=probs, y=class_labels, color="skyblue", ax=ax)
                ax.set_xlabel("Probability")
                # 不需要手動設定 yticklabels，seaborn 會自動處理
                st.pyplot(fig)


