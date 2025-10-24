import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 導入必要的函式庫 ---
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 【核心修正】: 設定 Matplotlib 以支援中文顯示 ---
# 註解: Streamlit 在後端使用 Matplotlib 繪製圖表，
#      我們需要手動指定一個支援中文的字體，否則中文會顯示為方塊。
#      'Microsoft JhengHei' 是 Windows 系統中常見的繁體中文字體。
#      如果您的系統沒有此字體，可以換成 'SimHei' (簡體) 或其他您安裝的中文字體。
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
except Exception as e:
    print(f"中文字體設定失敗: {e}")
    print("將使用預設字體，中文可能無法正常顯示。")


# --- 核心功能定義 ---

@st.cache_resource
def load_model():
    """
    載入您訓練好的 PyTorch 模型並設定為評估模式。
    """
    # 【最終修正】**: 將模型架構改回與您 Colab 訓練時一致的 EfficientNet-B0
    # 步驟 1: 重新定義模型架構
    model = models.efficientnet_b0(weights=None)  # 載入自己的權重(不使用預訓練權重)

    # 替換 EfficientNet-B0 的分類層 (classifier)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 7),
        nn.LogSoftmax(dim=1)
    )

    #  步驟 2: 載入訓練好的權重
    try:
        model.load_state_dict(torch.load('best_skin_cancer_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("錯誤：找不到模型權重檔案 'best_skin_cancer_model.pth'！")
        st.error("請確認您已將訓練好的模型權重檔案與此 app.py 放在同一個資料夾中。")
        return None

    # 步驟 3: 設定模型為評估模式
    model.eval()
    return model


def process_image(image_pil):
    """
    對 PIL 影像進行預處理，返回 PyTorch 張量。
    """
    # 【最終修正】**: 使用 EfficientNet 官方推薦的轉換流程
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
    0: 'Actinic keratoses (日光性角化)',
    1: 'Basal cell carcinoma (基底細胞癌)',
    2: 'Benign keratosis-like lesions (良性角化樣病變)',
    3: 'Dermatofibroma (皮膚纖維瘤)',
    4: 'Melanoma (惡性黑色素瘤)',
    5: 'Melanocytic nevi (黑色素細胞痣)',
    6: 'Vascular lesions (血管性病變)'
}
class_names = list(idx_to_lesion_name.values())

# 載入模型
model = load_model()

# 設定網頁標題
st.title("皮膚病灶AI輔助辨識系統")
st.write("請上傳一張皮膚病灶的圖片，AI模型將會為您預測可能的類型。")
st.caption("注意：系統預測結果僅供參考，請以主治醫師診斷為準。")

# 建立一個圖片上傳器
uploaded_file = st.file_uploader("請上傳一張欲進行輔助辨識的圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # 在網頁上顯示上傳的圖片
    # **【核心修正】**: 移除了過時的 'use_column_width=True' 參數。
    # 新版 Streamlit 會自動將圖片寬度調整為欄位寬度。
    st.image(image, caption='您上傳的圖片')
    st.write("")

    if st.button('開始預測', use_container_width=True):
        with st.spinner('模型正在分析中，請稍候...'):
            image_tensor = process_image(image)
            probs, classes_idx = predict(image_tensor, model)
            class_labels = [class_names[i] for i in classes_idx]
            pred_label_name = class_labels[0]

            st.subheader(f"AI系統預測結果： **{pred_label_name}**")
            st.write("前 5 名預測機率分佈：")
            fig, ax = plt.subplots()
            sns.barplot(x=probs, y=class_labels, color="skyblue", ax=ax)
            ax.set_xlabel("機率 (Probability)")
            # **【核心修正】**: 移除了下面這行多餘的程式碼。
            # Seaborn 會自動處理 Y 軸的標籤。
            # ax.set_yticklabels(class_labels, rotation=0)
            st.pyplot(fig)



