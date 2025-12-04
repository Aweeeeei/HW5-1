import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter
import jieba

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI/Human Detector Pro (Multi-lang)",
    page_icon="🇨🇳",
    layout="wide"
)

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定 (Settings)")
    lang_mode = st.radio(
        "選擇語言模式 (Language Mode)",
        ["Traditional Chinese (繁中)", "English"]
    )
    
    st.info("ℹ️ 中文模式使用 `jieba` 進行斷詞技術分析。")

st.title(f"📊 {lang_mode.split('(')[0]} 文本特徵分析器")
st.markdown("此工具透過統計學特徵（句長變異數、詞彙豐富度）輔助判斷是否為 AI 生成。")

# --- 停用詞設定 (過濾無意義詞彙) ---
STOPWORDS_EN = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'it', 'this', 'that'])
STOPWORDS_ZH = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '我們', '你們', '他們', '在', '這', '那'])

# --- 核心邏輯 ---
def analyze_text_features(text, mode):
    clean_text = text.strip()
    if not clean_text:
        return None

    sentences = []
    words = []
    filtered_words = []

    # === 針對不同語言的處理邏輯 ===
    if mode == "English":
        # 英文：用 . ! ? 切句，用空白切詞
        sentences = re.split(r'[.!?\n]+', clean_text)
        words = re.findall(r'\w+', clean_text.lower())
        stopwords = STOPWORDS_EN
        
    else: # Traditional Chinese
        # 中文：用 。 ！ ？ \n 切句
        sentences = re.split(r'[。！？\n]+', clean_text)
        # 使用 jieba 斷詞
        words = list(jieba.cut(clean_text))
        # 過濾掉標點符號與空白
        words = [w for w in words if w.strip() and len(w) > 0]
        stopwords = STOPWORDS_ZH

    # 移除空句子
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if len(words) < 5:
        return None

    # --- 特徵計算 (中英通用) ---
    
    # 句長計算 (中文算詞數，也可以改算字數，這裡統一算詞數/Token數)
    if mode == "English":
        sentence_lengths = [len(s.split()) for s in sentences]
    else:
        # 中文句長：計算該句切分後的詞數
        sentence_lengths = [len(list(jieba.cut(s))) for s in sentences]
    
    avg_len = np.mean(sentence_lengths)
    std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0

    # 詞彙豐富度
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # 過濾停用詞 (為了畫圖好看)
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1] # 中文通常過濾單字詞
    word_counts = Counter(filtered_words)

    # --- 評分邏輯 (Heuristic) ---
    score = 0.5 
    
    # 調整閾值：中文的斷句習慣跟英文略有不同，稍微寬鬆一點
    if std_dev < 4: score += 0.25      # 極度平穩 -> AI
    elif std_dev > 10: score -= 0.25   # 波動大 -> Human

    if ttr < 0.4: score += 0.15        # 用詞重複 -> AI
    elif ttr > 0.65: score -= 0.15     # 用詞豐富 -> Human

    final_score = min(max(score, 0.01), 0.99)
    
    return {
        "score": final_score,
        "sentences": sentences,
        "sentence_lengths": sentence_lengths,
        "avg_len": avg_len,
        "std_dev": std_dev,
        "ttr": ttr,
        "word_counts": word_counts,
        "total_words": len(words),
        "total_sentences": len(sentences)
    }

# --- UI 介面 ---
col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("📝 輸入區")
    placeholder_text = "請貼上中文文章..." if "Chinese" in lang_mode else "Paste English text here..."
    user_input = st.text_area("Input Text", height=300, placeholder=placeholder_text, label_visibility="collapsed")
    analyze_btn = st.button("🚀 開始深度分析", type="primary")

if analyze_btn and user_input:
    # 呼叫分析函數，傳入語言模式
    data = analyze_text_features(user_input, lang_mode)
    
    if data is None:
        st.warning("⚠️ 文本過短，無法分析。")
    else:
        with col_result:
            st.subheader("🔍 分析報告")
            
            ai_score = data['score']
            if ai_score > 0.6:
                result_text = "高度疑似 AI 生成"
                result_color = "red"
            elif ai_score < 0.4:
                result_text = "可能是 Human 撰寫"
                result_color = "green"
            else:
                result_text = "混合特徵 / 不確定"
                result_color = "orange"

            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background-color:rgba(128,128,128,0.1); border-left: 5px solid {result_color}">
                <h3 style="margin:0; color:{result_color}">{result_text} (AI 指數: {int(ai_score*100)}%)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("總句子數", data['total_sentences'])
            kpi2.metric("平均句長 (詞)", f"{data['avg_len']:.1f}")
            kpi3.metric("句長波動 (Std Dev)", f"{data['std_dev']:.1f}")
            kpi4.metric("詞彙豐富度 (TTR)", f"{data['ttr']:.2f}")

            tab1, tab2 = st.tabs(["📈 句型結構分析", "🔠 常用詞彙統計"])

            with tab1:
                st.caption("觀察重點：人類寫作時，句子長度（詞數）通常會有劇烈波動。")
                chart_data = pd.DataFrame({
                    "句子順序": range(1, len(data['sentence_lengths']) + 1),
                    "詞數": data['sentence_lengths']
                })
                st.line_chart(chart_data, x="句子順序", y="詞數", color="#FF4B4B")

            with tab2:
                st.caption("排除常見助詞（的、了、是...）後的關鍵字。")
                top_words = data['word_counts'].most_common(10)
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=["詞彙", "次數"])
                    st.bar_chart(words_df.set_index("詞彙"))
                else:
                    st.info("關鍵字數據不足")

elif not analyze_btn:
    with col_result:
        st.info("👈 請選擇語言模式，輸入文章並分析")