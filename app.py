import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter
import jieba
import zlib

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI/Human Detector Tuned",
    page_icon="⚖️",
    layout="wide"
)

# --- 範例資料庫 ---
EXAMPLES = {
    "English": [
        {
            "type": "AI",
            "text": "Artificial Intelligence involves the development of algorithms that allow computers to perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Machine learning, a subset of AI, focuses on building systems that learn from data. As technology advances, AI is becoming increasingly integrated into various sectors, including healthcare, finance, and transportation."
        },
        {
            "type": "Human",
            "text": "I literally cannot believe what just happened to me at the coffee shop! So, I ordered my usual latte, right? And the barista, who looked totally asleep, handed me a cup that felt way too light. I took a sip and—BAM—it was just hot milk! No coffee at all. Seriously? I just stood there laughing because, honestly, it's been that kind of week. Who forgets the coffee in a latte?"
        },
        {
            "type": "AI",
            "text": "Regular physical exercise is crucial for maintaining good overall health. It offers numerous benefits for both the body and the mind. Engaging in consistent physical activity helps to control body weight effectively. Furthermore, exercise can improve cardiovascular health significantly over time. Additionally, it strengthens muscles and bones, reducing the risk of injury. Regular physical exercise also boosts mental health by reducing stress and anxiety levels. Therefore, incorporating physical activity into one's daily routine is highly recommended for a healthy lifestyle."
        },
        {
            "type": "Human",
            "text": "You know that feeling when you finish a really good book and you just stare at the wall for like ten minutes? That was me last night. The ending was so unexpected, yet it made perfect sense. I was crying, smiling, just a total mess. I wish I could erase my memory and read it all over again for the first time. Truly a masterpiece."
        }
    ],
    "Traditional Chinese (繁中)": [
        {
            "type": "AI",
            "text": "區塊鏈技術是一種去中心化的分散式帳本技術，它確保了資料的透明性與不可篡改性。每一個區塊都包含了前一個區塊的加密雜湊值、時間戳記以及交易資料。這種結構使得區塊鏈在金融科技、供應鏈管理以及智慧合約等領域展現出巨大的應用潛力。隨著技術的成熟，我們預計將看到更多基於區塊鏈的創新解決方案。"
        },
        {
            "type": "Human",
            "text": "昨天跟朋友去排那家新開的拉麵店，真的排到天荒地老！我們在寒風中站了快兩個小時，腳都快斷了。結果進去一吃，哇賽，那個湯頭濃郁到不行，叉燒也是入口即化，瞬間覺得剛剛的辛苦都值得了。雖然這家店的價格有點小貴，但久久吃一次犒賞自己應該不過分吧？下次一定要挑平日來，不然真的會等到瘋掉。"
        },
        {
            "type": "AI",
            "text": "環境保護是當今全球面臨的一個重要議題。隨著工業化的快速發展，自然的生態平衡受到了嚴重的挑戰。我們必須意識到保護地球家園的緊迫性與必要性。減少一次性塑膠製品的使用是一個關鍵的步驟。此外，積極推廣再生能源的應用也是非常重要的措施。每個人都應該提高自身的環保意識並採取實際行動。只有透過共同的努力，我們才能實現可持續發展的長遠目標。"
        },
        {
            "type": "Human",
            "text": "救命啊！我刚刚把手機忘在計程車上了，現在整個人超焦慮。裡面有我所有的照片還有沒備份的聯絡人資料，如果找不回來我真的會崩潰。司機大哥也不接電話，客服又一直忙線中，這到底是甚麼倒楣的一天？拜託好心人如果撿到可以送去警察局，我願意請你吃大餐答謝，真的拜託了！"
        }
    ]
}

# --- Session State 初始化 ---
if 'input_text' not in st.session_state: st.session_state['input_text'] = ""
if 'example_index' not in st.session_state: st.session_state['example_index'] = 0

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 設定")
    lang_mode = st.radio("選擇語言模式", ["Traditional Chinese (繁中)", "English"])
    st.info("⚠️ 已啟用「高靈敏度模式」以加強 AI 偵測能力。")

st.title(f"⚖️ {lang_mode.split('(')[0]} 文本偵測器 (Tuned)")

STOPWORDS_EN = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'it', 'this', 'that'])
STOPWORDS_ZH = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '我們', '你們', '他們', '在', '這', '那'])

# --- 核心邏輯 (參數已調校) ---
def analyze_text_features(text, mode):
    clean_text = text.strip()
    if not clean_text: return None

    # 1. 斷詞斷句
    sentences, words, stopwords = [], [], []
    if mode == "English":
        sentences = re.split(r'[.!?\n]+', clean_text)
        words = re.findall(r'\w+', clean_text.lower())
        stopwords = STOPWORDS_EN
    else:
        sentences = re.split(r'[。！？\n]+', clean_text)
        words = list(jieba.cut(clean_text))
        words = [w for w in words if w.strip() and len(w) > 0]
        stopwords = STOPWORDS_ZH

    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(words) < 5: return None

    # 2. 計算特徵數值
    if mode == "English":
        sentence_lengths = [len(s.split()) for s in sentences]
    else:
        sentence_lengths = [len(list(jieba.cut(s))) for s in sentences]
    
    avg_len = np.mean(sentence_lengths)
    std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0

    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    
    text_bytes = clean_text.encode('utf-8')
    compressed_data = zlib.compress(text_bytes)
    compression_ratio = len(compressed_data) / len(text_bytes)

    # --- 3. 評分邏輯 (Tuned Thresholds) ---
    score_std = 0.5
    score_ttr = 0.5
    score_zlib = 0.5

    # [調整點 1] 放寬 Std Dev 判定：英文 7 以下都算平穩(AI)，中文 5 以下
    # 理由：現代 AI 比較會換句長了，所以要提高 AI 的容許範圍
    thresh_std_ai = 7.0 if mode == "English" else 5.0
    
    if std_dev < thresh_std_ai: 
        score_std = 1.0 # 強烈懷疑是 AI
    elif std_dev > (thresh_std_ai + 5): 
        score_std = 0.0 # Human
    else:
        # 中間地帶，稍微偏向 Human
        score_std = 0.4

    # [調整點 2] TTR 調整
    if ttr < 0.45: score_ttr = 1.0
    elif ttr > 0.65: score_ttr = 0.0
    else: score_ttr = 0.4

    # [調整點 3] Zlib 壓縮率調整 (最重要)
    # 短文本壓縮率會虛高，所以要放寬 AI 的上限
    # 英文：0.45 以下視為 AI (原本是 0.38)
    # 中文：0.55 以下視為 AI (原本是 0.43)
    thresh_zlib_ai = 0.45 if mode == "English" else 0.55
    
    if compression_ratio < thresh_zlib_ai: 
        score_zlib = 1.0
    elif compression_ratio > (thresh_zlib_ai + 0.1): 
        score_zlib = 0.0
    else:
        score_zlib = 0.4

    # 加權平均 (稍微降低 TTR 權重，因為短文 TTR 不準)
    final_score = (score_std * 0.35) + (score_ttr * 0.25) + (score_zlib * 0.40)
    
    return {
        "score": final_score, 
        "features": {
            "std_dev": std_dev,
            "ttr": ttr,
            "compression_ratio": compression_ratio,
            "thresh_std_ai": thresh_std_ai,     # 回傳閾值給 Debug 看
            "thresh_zlib_ai": thresh_zlib_ai    # 回傳閾值給 Debug 看
        },
        "sentence_lengths": sentence_lengths,
        "avg_len": avg_len, 
        "word_counts": Counter([w for w in words if w not in stopwords and len(w)>1]),
        "total_sentences": len(sentences)
    }

# --- UI ---
col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("📝 輸入區")
    
    def load_next_example():
        key = "English" if "English" in lang_mode else "Traditional Chinese (繁中)"
        examples = EXAMPLES[key]
        idx = st.session_state['example_index'] % len(examples)
        selected = examples[idx]
        st.session_state['input_text'] = selected['text']
        st.toast(f"已載入範例 #{idx+1} ({selected['type']})", icon="✅")
        st.session_state['example_index'] += 1

    st.button("🎲 載入範例 (輪播)", on_click=load_next_example, type="secondary")

    user_input = st.text_area("Input Text", height=350, key="input_text", placeholder="輸入文字...", label_visibility="collapsed")
    analyze_btn = st.button("🚀 開始分析", type="primary")

if analyze_btn and user_input:
    data = analyze_text_features(user_input, lang_mode)
    
    if data is None:
        st.warning("⚠️ 文本過短")
    else:
        with col_result:
            score = data['score']
            
            # 讓判定稍微嚴格一點： > 0.55 就算疑似 AI
            if score > 0.55:
                res_txt, res_color = "疑似 AI 生成", "red"
            elif score < 0.35:
                res_txt, res_color = "疑似 Human 撰寫", "green"
            else:
                res_txt, res_color = "混合特徵 / 不確定", "orange"

            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background-color:rgba(128,128,128,0.1); border-left: 6px solid {res_color}">
                <h3 style="margin:0; color:{res_color}">{res_txt}</h3>
                <p style="margin:5px 0 0 0; opacity:0.8">AI 可能性指數: <b>{int(score*100)}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Debug 區塊：這是你檢查為什麼「全部都判成 Human」的關鍵 ---
            with st.expander("🐞 開發者數據 (Debug Info)", expanded=True):
                f = data['features']
                st.write("如果數值 **小於** 閾值，會被判定為 AI。")
                
                c_d1, c_d2, c_d3 = st.columns(3)
                c_d1.metric("句長波動 (Std)", f"{f['std_dev']:.2f}", f"閾值: {f['thresh_std_ai']}")
                c_d2.metric("壓縮率 (Zlib)", f"{f['compression_ratio']:.2f}", f"閾值: {f['thresh_zlib_ai']}")
                c_d3.metric("詞彙豐富度", f"{f['ttr']:.2f}", "閾值: 0.45")
                
                st.caption(f"目前分數: {score:.2f} (0=Human, 1=AI)")

            # 圖表
            tab1, tab2 = st.tabs(["📈 句長波動", "🔠 詞彙統計"])
            with tab1:
                st.line_chart(pd.DataFrame({"Len": data['sentence_lengths']}), color="#FF4B4B")
            with tab2:
                top_words = data['word_counts'].most_common(10)
                if top_words: st.bar_chart(pd.DataFrame(top_words, columns=["W", "C"]).set_index("W"))