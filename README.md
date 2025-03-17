# 多模態生成式 AI 應用

一個基於 Streamlit 的 AI 應用，結合了圖像和文本輸入，使用 Gemma-3-27b-it 和 Flux-Schnell 模型生成創意內容。

## 功能特點

- 📤 支援圖片上傳（JPG、JPEG、PNG格式）
- ✍️ 可選的文字提示輸入
- 🎨 使用 Gemma-3-27b-it 生成文本描述
- 🖼️ 使用 Flux-Schnell 生成新圖像
- 🎯 可調節的生成參數
- 💫 美觀的用戶界面

## 安裝步驟

1. 克隆倉庫：
```bash
git clone [repository-url]
cd [repository-name]
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 配置環境變量：

複製 `.env.example` 文件（如果存在）或創建新的 `.env` 文件：
```bash
# Windows
copy .env.example .env
# 或直接創建 .env 文件
```

在 `.env` 文件中設置您的 Replicate API Token：
```
REPLICATE_API_TOKEN=your-token-here
```

您可以在 [Replicate](https://replicate.com/) 註冊並獲取 API Token。

4. 運行應用：
```bash
streamlit run app.py
```

## 使用說明

1. 打開瀏覽器訪問 `http://localhost:8501`
2. 上傳一張圖片
3. （可選）輸入文字提示
4. 調整生成參數：
   - 創造性程度：控制生成內容的創意程度
   - 圖像生成質量：控制生成圖像的細節程度
5. 點擊"開始生成"按鈕
6. 等待生成結果

## 注意事項

- 確保圖片大小適中，避免上傳過大的文件
- 生成過程可能需要一些時間，請耐心等待
- 確保網絡連接穩定

## 技術棧

- Streamlit：用戶界面框架
- Replicate API：AI 模型調用
- Gemma-3-27b-it：多模態文本生成模型
- Flux-Schnell：圖像生成模型
- Pillow：圖像處理庫
