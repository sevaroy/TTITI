import streamlit as st
import replicate
import os
from PIL import Image
import io
from dotenv import load_dotenv, find_dotenv
import base64
import tempfile

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(
    page_title="多模態生成式 AI 應用",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

# Check Replicate API Token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    st.error("⚠️ 請設定 REPLICATE_API_TOKEN 環境變數！")
    st.code("export REPLICATE_API_TOKEN='your-token-here'")
    st.stop()

# Initialize Replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# App title and description
st.title("🎨 多模態生成式 AI 應用")
st.markdown("""
使用 Gemma-3-27b-it 和 Flux-Schnell 模型，將您的圖片和文字轉化為新的創意作品。
- 📤 上傳一張圖片
- ✍️ 添加文字提示（可選）
- 🎯 調整生成參數
- 🪄 獲得 AI 生成的文本描述和新圖像
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    # Image upload
    uploaded_file = st.file_uploader(
        "上傳圖片",
        type=["jpg", "jpeg", "png"],
        help="支援 JPG、JPEG 和 PNG 格式"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="上傳的圖片", use_container_width=True)

with col2:
    # Text prompt
    prompt = st.text_area(
        "文字提示（可選）",
        value="描述這張圖片並想象一個未來的場景。",
        help="添加額外的文字提示來引導 AI 生成"
    )
    
    # Generation parameters
    temperature = st.slider(
        "創造性程度",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        help="較高的值會產生更有創意但可能不太相關的結果"
    )
    
    num_inference_steps = st.slider(
        "圖像生成質量",
        min_value=1,
        max_value=4,
        value=3,
        help="步數越多，生成的圖像質量越高，但需要更長時間"
    )

# Generate button
if st.button("🚀 開始生成", type="primary"):
    if uploaded_file is None:
        st.error("請先上傳一張圖片！")
    else:
        try:
            with st.spinner("🤖 AI 正在創作中..."):
                # Prepare image for API
                image = Image.open(uploaded_file)
                
                # Save image to a temporary file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "temp_image.png")
                image.save(temp_path, format="PNG")

                # Generate text using Gemma
                text_output = client.run(
                    "google-deepmind/gemma-3-27b-it:c0f0aebe8e578c15a7531e08a62cf01206f5870e9d0a67804b8152822db58c54",
                    input={
                        "image": open(temp_path, "rb"),
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_new_tokens": 512,
                        "top_p": 0.95,
                    }
                )
                generated_text = "".join(text_output)
                st.session_state.generated_text = generated_text

                # Generate image using Flux-Schnell
                image_output = client.run(
                    "black-forest-labs/flux-schnell",
                    input={
                        "prompt": generated_text,
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "png",
                        "num_inference_steps": num_inference_steps,
                    }
                )
                st.session_state.generated_image = image_output[0]

                # Clean up temporary file
                os.remove(temp_path)
                os.rmdir(temp_dir)

            # Display results
            st.success("✨ 生成完成！")
            
            # Show generated text
            st.subheader("📝 生成的文本描述")
            st.write(st.session_state.generated_text)
            
            # Show generated image
            st.subheader("🖼️ 生成的新圖像")
            st.image(
                st.session_state.generated_image,
                caption="AI 創作的新圖像",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"生成過程中出現錯誤: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>基於 Gemma-3-27b-it 和 Flux-Schnell 模型構建 | 使用 Replicate API</p>
</div>
""", unsafe_allow_html=True)
