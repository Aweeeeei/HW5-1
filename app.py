import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter
import jieba
import zlib  # <--- 新增核心：用於計算資訊熵 (壓縮率)

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI/Human Detector Ultra",
    page_icon="🧬",
    layout="wide"
)

# --- 定義範例資料庫 (包含之前的擴充範例) ---
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

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定 (Settings)")
    lang_mode = st.radio("選擇語言模式", ["Traditional Chinese (繁中)", "English"])
    st.markdown("---")
    st.info("""
    **🧬 Ultra 核心技術：**
    除了句法分析外，此版本引入 **Zlib 壓縮算法** 來計算「文本熵」。
    - **原理**：AI 生成的文本通常規律性較強，壓縮率較高（檔案變小）。
    - **權重**：熵值佔評分的 40%。
    """)

st.title(f"🧬 {lang_mode.split('(')[0]} 文本偵測器 (Ultra版)")

# --- 停用詞 ---
STOPWORDS_EN = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'it', 'this', 'that'])
STOPWORDS_ZH = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '我們', '你們', '他們', '在', '這', '那'])

# --- 核心邏輯：加入 Zlib 演算法 ---
def analyze_text_features(text, mode):
    clean_text = text.strip()
    if not clean_text: return None

    # 1. 基礎前處理
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

    # 2. 特徵 A：句長波動 (Burstiness)
    if mode == "English":
        sentence_lengths = [len(s.split()) for s in sentences]
    else:
        sentence_lengths = [len(list(jieba.cut(s))) for s in sentences]
    
    avg_len = np.mean(sentence_lengths)
    std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0

    # 3. 特徵 B：詞彙豐富度 (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
    word_counts = Counter(filtered_words)

    # 4. 特徵 C：資訊熵 / 壓縮率 (Zlib Entropy) [NEW]
    # 將文本轉為 bytes 並壓縮，計算壓縮比率
    text_bytes = clean_text.encode('utf-8')
    compressed_data = zlib.compress(text_bytes)
    compression_ratio = len(compressed_data) / len(text_bytes)

    # --- 綜合加權評分系統 ---
    # 目標：將各項指標轉換為 0 (Human) ~ 1 (AI) 的分數
    
    # (A) 句長評分 (30%)
    score_std = 0.5
    thresh_std_low = 5 if mode == "English" else 3
    thresh_std_high = 12 if mode == "English" else 10
    
    if std_dev < thresh_std_low: score_std = 1.0     # AI (平穩)
    elif std_dev > thresh_std_high: score_std = 0.0  # Human (波動)

    # (B) 豐富度評分 (30%)
    score_ttr = 0.5
    if ttr < 0.4: score_ttr = 1.0        # AI (重複)
    elif ttr > 0.65: score_ttr = 0.0     # Human (豐富)

    # (C) 壓縮率評分 (40%) [最關鍵指標]
    score_zlib = 0.5
    # 根據經驗法則設定的閾值
    thresh_zlib_ai = 0.38 if mode == "English" else 0.43
    thresh_zlib_human = 0.50 if mode == "English" else 0.55
    
    if compression_ratio < thresh_zlib_ai: score_zlib = 1.0      # AI (規律好壓)
    elif compression_ratio > thresh_zlib_human: score_zlib = 0.0 # Human (混亂難壓)

    # 計算加權平均分
    final_score = (score_std * 0.3) + (score_ttr * 0.3) + (score_zlib * 0.4)
    
    return {
        "score": final_score, 
        "features": {
            "std_dev": std_dev,
            "ttr": ttr,
            "compression_ratio": compression_ratio
        },
        "sentences": sentences, 
        "sentence_lengths": sentence_lengths,
        "avg_len": avg_len, 
        "word_counts": word_counts,
        "total_sentences": len(sentences)
    }

# --- UI 介面 ---
col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("📝 輸入區")
    
    # --- 🎲 範例按鈕 (保持你的功能) ---
    def load_next_example():
        key = "English" if "English" in lang_mode else "Traditional Chinese (繁中)"
        examples = EXAMPLES[key]
        idx = st.session_state['example_index'] % len(examples)
        selected = examples[idx]
        st.session_state['input_text'] = selected['text']
        st.toast(f"已載入範例 #{idx+1} ({selected['type']})", icon="✅")
        st.session_state['example_index'] += 1

    st.button("🎲 載入範例 (輪播)", on_click=load_next_example, type="secondary")

    user_input = st.text_area(
        "Input Text",
        height=350, 
        placeholder="請輸入文字...", 
        label_visibility="collapsed",
        key="input_text" 
    )
    
    analyze_btn = st.button("🚀 開始深度分析", type="primary")

# --- 分析結果顯示 ---
if analyze_btn and user_input:
    data = analyze_text_features(user_input, lang_mode)
    
    if data is None:
        st.warning("⚠️ 文本過短，無法分析。")
    else:
        with col_result:
            st.subheader("🔍 分析報告")
            
            score = data['score']
            if score > 0.65:
                res_txt, res_color = "高度疑似 AI 生成", "red"
            elif score < 0.35:
                res_txt, res_color = "可能是 Human 撰寫", "green"
            else:
                res_txt, res_color = "混合特徵 / 不確定", "orange"

            st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background-color:rgba(128,128,128,0.1); border-left: 6px solid {res_color}">
                <h3 style="margin:0; color:{res_color}">{res_txt}</h3>
                <p style="margin:5px 0 0 0; opacity:0.8">AI 可能性指數: <b>{int(score*100)}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")

            # --- 3個關鍵指標 Dashboard (新增壓縮率) ---
            f = data['features']
            c1, c2, c3 = st.columns(3)
            
            c1.metric("1. 句長波動度", f"{f['std_dev']:.1f}", 
                      delta="低 (像AI)" if f['std_dev'] < 5 else "高 (像人)", delta_color="inverse")
            
            c2.metric("2. 詞彙豐富度", f"{f['ttr']:.2f}",
                      delta="低 (像AI)" if f['ttr'] < 0.4 else "高 (像人)", delta_color="inverse")
            
            c3.metric("3. 資訊熵 (壓縮率)", f"{f['compression_ratio']:.2f}",
                      delta="低 (像AI)" if f['compression_ratio'] < 0.4 else "高 (像人)", delta_color="inverse",
                      help="數值越低代表文本越規律、越容易被預測 (AI特徵)")

            # --- 圖表區 ---
            tab1, tab2 = st.tabs(["📈 句型結構分析", "🔠 常用詞彙統計"])

            with tab1:
                st.caption("Human 通常句長波動大 (線條劇烈跳動)；AI 則較平穩。")
                chart_data = pd.DataFrame({
                    "句序": range(1, len(data['sentence_lengths']) + 1),
                    "詞數": data['sentence_lengths']
                })
                st.line_chart(chart_data, x="句序", y="詞數", color="#FF4B4B")

            with tab2:
                top_words = data['word_counts'].most_common(10)
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=["詞彙", "次數"])
                    st.bar_chart(words_df.set_index("詞彙"))
                else:
                    st.info("關鍵字數據不足")

elif not analyze_btn:
    with col_result:
        st.info("👈 點擊「🎲 載入範例」測試最新的多維度偵測演算法。")