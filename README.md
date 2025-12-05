# 🤖 AI / Human Text Detector (Homework 5-1)

一個基於 **Streamlit** 的輕量化 AI 文本偵測工具。本專案採用 **統計學特徵萃取 (Statistical Feature Extraction)** 技術，能快速分析文本是否由 AI 生成，並提供視覺化的數據儀表板。

### 為什麼選擇「自建特徵」而非「預訓練模型」？

雖然使用 Hugging Face 的預訓練模型 (如 `roberta-base-openai-detector`) 準確度較高，但本專案選擇使用 **自建特徵 (Feature Extraction)** 方法，主要基於以下 **Streamlit Cloud 部署限制** 與 **效能考量**：

1.  **記憶體限制 (RAM Constraints)**：
    * Streamlit Cloud 的免費版資源有限。載入一個數百 MB 的 Transformer 模型容易導致 **OOM (Out of Memory)** 崩潰或啟動極慢。
    * 自建特徵法僅依賴 `numpy` 與 `zlib`，記憶體佔用極低。
2.  **冷啟動速度 (Cold Start Latency)**：
    * 預訓練模型在每次部署或重啟時需下載 500MB+ 的權重檔，使用者需等待許久。
    * 本專案採用演算法計算，**網頁秒開，分析結果即時顯示**。
3.  **依賴單純**：
    * 無需安裝 `torch` 或 `tensorflow` 等龐大的深度學習框架，環境建置輕量且穩定。

### 偵測原理
我們結合了三個核心指標進行加權評分：
1.  **Burstiness (句長標準差)**：人類寫作時長短句交錯明顯；AI 則傾向於結構工整。
2.  **Perplexity Proxy (詞彙多樣性)**：計算 Type-Token Ratio (TTR)。
3.  **Text Entropy (壓縮率)**：AI 生成的文本基於機率模型，規律性強，因此 `zlib` 壓縮率通常較低（檔案變小）。

## 📖 使用說明 (How to Use)

1.  **選擇語言**：在左側側邊欄選擇「Traditional Chinese」或「English」。
2.  **輸入文本**：
    * 你可以直接在輸入框貼上文章。
    * 或者點擊 **「🎲 載入範例」** 按鈕，系統會自動輪播 AI 與人類的測試文章。
3.  **開始分析**：點擊 **「🚀 開始分析」** 按鈕。
4.  **查看報告**：右側將顯示 AI 可能性指數、關鍵指標儀表板以及可視化圖表。

## 🛠️ 安裝與執行 (Local Installation)

直接前往：https://aiot-hw5-1.streamlit.app/

如果你想在本地端執行此專案：

1.  **Clone 專案**
    ```bash
    git clone <your-repo-url>
    cd <your-project-folder>
    ```

2.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```
    *(requirements.txt 包含: streamlit, numpy, pandas, jieba)*

3.  **啟動 Streamlit**
    ```bash
    streamlit run app.py
    ```

---

**Developed for Homework 5-1**