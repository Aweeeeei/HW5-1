import streamlit as st
from transformers import pipeline
import time

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI vs Human Detector",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI 文本偵測器 (AI Detector)")
st.write("這是一個基於 Transformer 模型的簡單檢測工具，用於判斷文本是由 **人類** 還是 **AI** 撰寫的。")

# --- 1. 載入模型 (使用 @st.cache_resource 避免重複載入) ---
@st.cache_resource
def load_detector_model():
    # 這裡使用 roberta-base-openai-detector (經典模型)
    # 注意：首次執行會下載模型 (約 500MB)，請耐心等待
    model_name = "roberta-base-openai-detector"
    classifier = pipeline("text-classification", model=model_name)
    return classifier

# 顯示載入狀態
with st.spinner("正在載入 AI 偵測模型..."):
    try:
        classifier = load_detector_model()
    except Exception as e:
        st.error(f"模型載入失敗，請檢查網路連接或套件安裝: {e}")
        st.stop()

# --- 2. UI 介面 ---
st.markdown("### 📝 請輸入要檢測的文本")
user_input = st.text_area(
    "在這裡貼上文章內容 (建議輸入 50 字以上)...",
    height=200,
    placeholder="Once upon a time, in a land far away..."
)

# --- 3. 偵測邏輯 ---
if st.button("開始偵測 (Analyze)", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ 請先輸入內容再進行偵測！")
    else:
        # 顯示處理中的狀態
        progress_text = "正在分析文本特徵..."
        my_bar = st.progress(0, text=progress_text)
        
        # 模擬一點進度條動畫 (讓 UI 感覺更順暢)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # 使用模型進行預測
        # 模型會回傳 [{'label': 'Fake', 'score': 0.99...}] 或 [{'label': 'Real', 'score': ...}]
        # 在此模型中，'Fake' = AI 生成, 'Real' = 人類撰寫
        result = classifier(user_input)[0]
        
        my_bar.empty() # 清除進度條

        # --- 4. 結果解析與視覺化 ---
        label = result['label']
        score = result['score']
        
        # 轉換邏輯：計算 AI 的機率與 Human 的機率
        if label == 'Fake':
            ai_prob = score
            human_prob = 1 - score
        else:
            human_prob = score
            ai_prob = 1 - score
            
        # 轉換成百分比
        ai_percent = ai_prob * 100
        human_percent = human_prob * 100

        # --- 顯示結果區塊 ---
        st.markdown("---")
        st.subheader("📊 檢測結果 (Analysis Result)")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="🤖 AI 可能性", value=f"{ai_percent:.1f}%")
            st.progress(ai_prob) # 顯示 AI 進度條

        with col2:
            st.metric(label="🧑 Human 可能性", value=f"{human_percent:.1f}%")
            # Human 進度條 (為了視覺區隔，你可以選擇不顯示或用不同顏色，Streamlit 預設同色)
            st.progress(human_prob)

        # 最終判斷結論
        st.markdown("### 結論：")
        if ai_prob > 0.5:
            st.error(f"這篇文章 **{ai_percent:.1f}%** 像是由 AI 生成的。")
        else:
            st.success(f"這篇文章 **{human_percent:.1f}%** 像是由人類撰寫的。")

# --- 側邊欄資訊 ---
with st.sidebar:
    st.info("ℹ️ 關於此工具")
    st.markdown("""
    - **模型來源**: huggingface/roberta-base-openai-detector
    - **原理**: 使用預訓練的 Transformer 架構分析語意特徵與困惑度 (Perplexity)。
    - **限制**: 對於最新的 GPT-4 模型可能準確度會下降。
    """)