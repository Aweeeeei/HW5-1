import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI/Human Detector Pro",
    page_icon="📊",
    layout="wide" # 改為寬螢幕模式以容納圖表
)

st.title("📊 AI vs Human 文本特徵分析器")
st.markdown("此工具透過統計學特徵（句長變異數、詞彙豐富度）將文本「視覺化」，以輔助判斷是否為 AI 生成。")

# --- 簡單的停用詞表 (為了過濾掉 the, a, is 這種無意義詞) ---
STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'it', 'this', 'that'
])

# --- 核心邏輯：特徵提取與分析 ---
def analyze_text_features(text):
    clean_text = text.strip()
    if not clean_text:
        return None

    # 1. 切分句子
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # 2. 切分單字
    words = re.findall(r'\w+', clean_text.lower())
    
    if len(words) < 5:
        return None

    # --- 特徵計算 ---
    # 句長列表
    sentence_lengths = [len(s.split()) for s in sentences]
    
    # 平均句長與標準差
    avg_len = np.mean(sentence_lengths)
    std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0

    # 詞彙豐富度 (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # 過濾後的詞頻 (移除停用詞)
    filtered_words = [w for w in words if w not in STOPWORDS]
    word_counts = Counter(filtered_words)

    # --- 評分邏輯 ---
    score = 0.5 
    # AI 傾向於標準差小 (平穩)
    if std_dev < 6: score += 0.25
    elif std_dev > 12: score -= 0.25 # Human 傾向於標準差大 (波動)

    # AI 傾向於豐富度低 (重複)
    if ttr < 0.45: score += 0.15
    elif ttr > 0.65: score -= 0.15

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
col_input, col_result = st.columns([1, 2]) # 左窄右寬

with col_input:
    st.subheader("📝 輸入區")
    user_input = st.text_area(
        "請貼上英文文章",
        height=300,
        placeholder="貼上你的文章..."
    )
    analyze_btn = st.button("🚀 開始深度分析", type="primary")

# --- 分析結果顯示 ---
if analyze_btn and user_input:
    data = analyze_text_features(user_input)
    
    if data is None:
        st.warning("⚠️ 文本過短，無法進行有效統計分析。")
    else:
        with col_result:
            st.subheader("🔍 分析報告")
            
            # 1. 頂部結果卡片
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
            
            st.write("") # Spacer

            # 2. 關鍵指標 (KPIs)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("總句子數", data['total_sentences'])
            kpi2.metric("平均句長 (字)", f"{data['avg_len']:.1f}")
            kpi3.metric("句長波動 (Std Dev)", f"{data['std_dev']:.1f}", help="數值越高代表長短句交錯越明顯 (Human特徵)")
            kpi4.metric("詞彙豐富度 (TTR)", f"{data['ttr']:.2f}", help="數值越高代表用詞越不重複")

            # 3. 分頁顯示圖表
            tab1, tab2, tab3 = st.tabs(["📈 句型結構分析", "🔠 常用詞彙統計", "📄 原始數據"])

            with tab1:
                st.markdown("**句長波動圖 (Sentence Burstiness)**")
                st.caption("AI 通常像機器人一樣規律 (線條平緩)，人類寫作則情緒起伏大 (線條劇烈跳動)。")
                
                # 建立 DataFrame 給圖表用
                chart_data = pd.DataFrame({
                    "句子順序": range(1, len(data['sentence_lengths']) + 1),
                    "句子長度 (單字數)": data['sentence_lengths']
                })
                
                st.line_chart(
                    chart_data, 
                    x="句子順序", 
                    y="句子長度 (單字數)",
                    color="#FF4B4B"
                )

            with tab2:
                st.markdown("**高頻詞彙 (Top Keywords)**")
                st.caption("排除常見介系詞後的關鍵字分佈。")
                
                # 取出前 10 名
                top_words = data['word_counts'].most_common(10)
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=["單字", "出現次數"])
                    st.bar_chart(words_df.set_index("單字"))
                else:
                    st.info("沒有足夠的關鍵字資料。")

            with tab3:
                st.json({
                    "AI_Score": data['score'],
                    "Sentence_Lengths": data['sentence_lengths'],
                    "Sentences": data['sentences']
                })

elif not analyze_btn:
    with col_result:
        st.info("👈 請在左側輸入文章並按下分析按鈕")