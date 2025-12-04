import streamlit as st
import numpy as np
import re

# --- 頁面設定 ---
st.set_page_config(
    page_title="Lightweight AI Detector",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ 輕量版 AI 文本偵測器")
st.write("此版本使用 **統計特徵 (Statistical Features)** 進行分析，無需下載大型模型，執行速度極快。")

# --- 核心邏輯：自建特徵算法 ---
def analyze_text_features(text):
    """
    這是一個啟發式的算法 (Heuristic)，模擬 AI 偵測的邏輯：
    1. AI 生成的文章通常句型結構比較「平穩」，標準差 (Burstiness) 較低。
    2. 人類寫的文章通常會有長短句交錯，變化較大。
    3. 重複詞彙率：AI 有時會為了通順而使用常見詞。
    """
    
    # 1. 預處理
    clean_text = text.strip()
    if not clean_text:
        return 0.5, "無法判斷"

    # 切分句子 (簡單用 . ! ? 切分)
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # 切分單字
    words = re.findall(r'\w+', clean_text.lower())
    
    if len(words) < 5:
        return 0.5, "文本過短"

    # --- 特徵 1: 句長變異數 (Burstiness) ---
    # 人類寫作時，句子的長度變化通常較大 (有長有短)
    sentence_lengths = [len(s.split()) for s in sentences]
    if len(sentence_lengths) > 1:
        std_dev = np.std(sentence_lengths)
    else:
        std_dev = 0  # 只有一句話

    # --- 特徵 2: 詞彙豐富度 (Type-Token Ratio) ---
    # 人類通常會使用比較多樣的詞彙
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # --- 綜合評分算法 (模擬邏輯) ---
    # 基礎分數 50%
    # 變異數大 (Human) -> 分數往 0 (Human) 扣
    # 變異數小 (AI)    -> 分數往 1 (AI) 加
    
    score = 0.5 # 0=Human, 1=AI
    
    # 權重調整 (這些數值是經驗法則，為了作業演示用)
    if std_dev < 5: 
        score += 0.3  # 句長太規律，像 AI
    elif std_dev > 10:
        score -= 0.3  # 句長變化劇烈，像 Human

    if ttr < 0.4:
        score += 0.2  # 重複用詞多，像 AI (或品質差的文本)
    elif ttr > 0.7:
        score -= 0.1  # 用詞豐富，傾向 Human

    # 限制分數在 0.01 ~ 0.99 之間
    final_score = min(max(score, 0.01), 0.99)
    
    return final_score, f"句長標準差: {std_dev:.1f} | 詞彙豐富度: {ttr:.2f}"

# --- UI 介面 ---
st.markdown("### 📝 請輸入要檢測的文本")
user_input = st.text_area(
    "輸入英文文章進行測試...",
    height=200,
    placeholder="Artificial Intelligence is a fascinating field..."
)

if st.button("開始分析 (Analyze)", type="primary"):
    if not user_input.strip():
        st.warning("請輸入內容！")
    else:
        # 執行分析
        ai_prob, debug_info = analyze_text_features(user_input)
        human_prob = 1.0 - ai_prob
        
        # 顯示詳細數據 (Optional，讓作業看起來更專業)
        with st.expander("查看統計特徵數據"):
            st.code(debug_info)

        # 結果視覺化
        st.subheader("📊 分析結果")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 AI 可能性", f"{ai_prob*100:.1f}%")
            st.progress(ai_prob)
            
        with col2:
            st.metric("🧑 Human 可能性", f"{human_prob*100:.1f}%")
            st.progress(human_prob)

        # 結論
        st.markdown("---")
        if ai_prob > 0.6:
            st.error(f"判定結果：**高度疑似 AI 生成**")
            st.write("理由：句型結構過於工整，缺乏人類寫作的隨機性。")
        elif ai_prob < 0.4:
            st.success(f"判定結果：**可能是人類撰寫**")
            st.write("理由：句型長短變化自然，詞彙使用多樣。")
        else:
            st.info(f"判定結果：**不確定 / 混合內容**")
            st.write("特徵不明顯，可能篇幅過短或包含混合特徵。")