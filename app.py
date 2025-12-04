import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter
import jieba

# --- 頁面設定 ---
st.set_page_config(
    page_title="AI/Human Detector Pro",
    page_icon="🤖",
    layout="wide"
)

# --- 定義範例資料庫 (中英文各 2 AI / 2 Human) ---
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
            "text": "Climate change refers to significant changes in global temperature and weather patterns over time. While climate change is a natural phenomenon, scientific evidence suggests that human activities, particularly the burning of fossil fuels, are the primary drivers of recent warming trends. This leads to rising sea levels, more frequent extreme weather events, and disruptions to ecosystems."
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
            "text": "光合作用是植物、藻類和某些細菌利用陽光將二氧化碳和水轉化為葡萄糖和氧氣的過程。這個過程對於地球上的生命至關重要，因為它不僅提供了食物鏈的基礎能量，還釋放了生物呼吸所需的氧氣。光合作用主要發生在葉綠體的類囊體膜上，涉及光反應和暗反應兩個階段。"
        },
        {
            "type": "Human",
            "text": "救命啊！我刚刚把手機忘在計程車上了，現在整個人超焦慮。裡面有我所有的照片還有沒備份的聯絡人資料，如果找不回來我真的會崩潰。司機大哥也不接電話，客服又一直忙線中，這到底是甚麼倒楣的一天？拜託好心人如果撿到可以送去警察局，我願意請你吃大餐答謝，真的拜託了！"
        }
    ]
}

# --- Session State 初始化 ---
# 我們需要記住兩個變數：
# 1. input_text: 輸入框目前的內容
# 2. example_index: 目前輪播到第幾個範例
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""
if 'example_index' not in st.session_state:
    st.session_state['example_index'] = 0

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定 (Settings)")
    # 注意：這裡加上 key，讓 streamlit 自動更新變數
    lang_mode = st.radio(
        "選擇語言模式 (Language Mode)",
        ["Traditional Chinese (繁中)", "English"]
    )
    st.info("ℹ️ 中文模式使用 `jieba` 斷詞；英文模式使用空白切分。")

st.title(f"📊 {lang_mode.split('(')[0]} 文本分析器")

# --- 停用詞 ---
STOPWORDS_EN = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at', 'to', 'it', 'this', 'that'])
STOPWORDS_ZH = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '我們', '你們', '他們', '在', '這', '那'])

# --- 核心邏輯 (維持不變) ---
def analyze_text_features(text, mode):
    clean_text = text.strip()
    if not clean_text: return None

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

    if mode == "English":
        sentence_lengths = [len(s.split()) for s in sentences]
    else:
        sentence_lengths = [len(list(jieba.cut(s))) for s in sentences]
    
    avg_len = np.mean(sentence_lengths)
    std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0

    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
    word_counts = Counter(filtered_words)

    score = 0.5 
    if std_dev < 4: score += 0.25      # AI (平穩)
    elif std_dev > 10: score -= 0.25   # Human (波動)
    if ttr < 0.4: score += 0.15        # AI (重複)
    elif ttr > 0.65: score -= 0.15     # Human (豐富)
    final_score = min(max(score, 0.01), 0.99)
    
    return {
        "score": final_score, "sentences": sentences, "sentence_lengths": sentence_lengths,
        "avg_len": avg_len, "std_dev": std_dev, "ttr": ttr, "word_counts": word_counts,
        "total_sentences": len(sentences)
    }

# --- UI 介面 ---
col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("📝 輸入區")
    
    # --- 🎲 範例按鈕區塊 ---
    # 使用 callback 函數來更新 session state，避免邏輯混亂
    def load_next_example():
        # 決定目前的語言 key
        dict_key = "English" if "English" in lang_mode else "Traditional Chinese (繁中)"
        examples = EXAMPLES[dict_key]
        
        # 取得目前的 index
        idx = st.session_state['example_index'] % len(examples)
        
        # 更新輸入框文字
        selected_example = examples[idx]
        st.session_state['input_text'] = selected_example['text']
        
        # 顯示 Toast 提示 (短暫出現的訊息)
        st.toast(f"已載入範例 #{idx+1}", icon="✅")
        
        # Index + 1 準備下一次
        st.session_state['example_index'] += 1

    st.button("🎲 載入範例", on_click=load_next_example, type="secondary")

    # --- 文字輸入框 ---
    # 這裡將 key 綁定到 'input_text'，這樣按鈕更新 state 時，這裡會自動變
    user_input = st.text_area(
        "Input Text",
        height=300, 
        placeholder="請輸入文字或點擊上方範例按鈕...", 
        label_visibility="collapsed",
        key="input_text" 
    )
    
    analyze_btn = st.button("🚀 開始深度分析", type="primary")

# --- 分析結果顯示 (維持不變) ---
if analyze_btn and user_input:
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
            kpi2.metric("平均句長", f"{data['avg_len']:.1f}")
            kpi3.metric("句長波動 (Std Dev)", f"{data['std_dev']:.1f}")
            kpi4.metric("詞彙豐富度 (TTR)", f"{data['ttr']:.2f}")

            tab1, tab2 = st.tabs(["📈 句型結構分析", "🔠 常用詞彙統計"])

            with tab1:
                st.caption("Human 通常句長波動大 (線條劇烈跳動)；AI 則較平穩。")
                chart_data = pd.DataFrame({
                    "句子順序": range(1, len(data['sentence_lengths']) + 1),
                    "詞數": data['sentence_lengths']
                })
                st.line_chart(chart_data, x="句子順序", y="詞數", color="#FF4B4B")

            with tab2:
                top_words = data['word_counts'].most_common(10)
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=["詞彙", "次數"])
                    st.bar_chart(words_df.set_index("詞彙"))
                else:
                    st.info("關鍵字數據不足")

elif not analyze_btn:
    with col_result:
        st.info("👈 點擊「🎲 載入範例」快速體驗功能，或自行輸入文章。")