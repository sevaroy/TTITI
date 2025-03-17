import streamlit as st
import replicate
import os
from PIL import Image
import io
import tempfile
from dotenv import load_dotenv, find_dotenv
import base64
import uuid
import time

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set page config
st.set_page_config(
    page_title="多模態生成式 AI 應用",
    page_icon="🎨",
    layout="wide"
)

# ----------------- AI Service Functions -----------------

class AIService:
    def __init__(self, api_token):
        self.client = replicate.Client(api_token=api_token)
        
    def generate_text_from_image(self, image_path, prompt, temperature=0.7):
        """使用 Gemma-3-27b-it 從圖片生成文本"""
        try:
            text_output = self.client.run(
                "google-deepmind/gemma-3-27b-it:c0f0aebe8e578c15a7531e08a62cf01206f5870e9d0a67804b8152822db58c54",
                input={
                    "image": open(image_path, "rb"),
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": 512,
                    "top_p": 0.95,
                }
            )
            return "".join(text_output)
        except Exception as e:
            st.error(f"文本生成失敗: {str(e)}")
            return None
    
    def generate_image_from_text(self, text_prompt, num_outputs=1, num_inference_steps=3, aspect_ratio="1:1"):
        """使用 Flux-Schnell 從文本生成圖片"""
        try:
            image_output = self.client.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": text_prompt,
                    "num_outputs": num_outputs,
                    "aspect_ratio": aspect_ratio,
                    "output_format": "png",
                    "num_inference_steps": num_inference_steps,
                }
            )
            return image_output
        except Exception as e:
            st.error(f"圖像生成失敗: {str(e)}")
            return None

# ----------------- Session State Initialization -----------------

def init_session_state():
    if 'ai_service' not in st.session_state:
        # Check Replicate API Token
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            st.error("⚠️ 請設定 REPLICATE_API_TOKEN 環境變數！")
            st.code("export REPLICATE_API_TOKEN='your-token-here'")
            st.stop()
        st.session_state.ai_service = AIService(api_token)
    
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    
    if 'image_analyses' not in st.session_state:
        st.session_state.image_analyses = {}
    
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = {}

# ----------------- UI Components -----------------

def render_header():
    st.title("🎨 多模態生成式 AI 應用")
    st.markdown("""
    使用 Gemma-3-27b-it 和 Flux-Schnell 模型，將您的圖片和文字轉化為新的創意作品。
    - 📤 上傳多張圖片進行分析
    - ✍️ 添加文字提示（可選）
    - 🎯 調整生成參數
    - 🎲 選擇要生成的圖片數量
    - 🪄 獲得 AI 生成的文本描述和新圖像
    - 🎭 LINE 貼圖模式：生成適合作為 LINE 貼圖的圖片
    """)

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 生成參數")
        
        # Add LINE sticker mode toggle
        line_sticker_mode = st.toggle(
            "LINE 貼圖模式",
            value=True,
            help="啟用後，生成的圖片將更適合作為 LINE 貼圖使用"
        )
        
        prompt = st.text_area(
            "文字提示（可選）",
            value="描述這張圖片並想象一個未來的場景。" if not line_sticker_mode else "將這張圖片轉換為可愛、表情豐富的 LINE 貼圖風格。",
            help="添加額外的文字提示來引導 AI 生成"
        )
        
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
        
        num_images = st.slider(
            "生成圖片數量",
            min_value=1,
            max_value=4,
            value=1,
            help="為每張上傳的圖片生成指定數量的新圖片"
        )
        
        # Modify aspect ratio options based on LINE sticker mode
        if line_sticker_mode:
            aspect_ratio = "1:1"  # LINE stickers are typically square
            st.info("LINE 貼圖模式已啟用，圖片比例已設為 1:1（正方形）")
        else:
            aspect_ratio = st.selectbox(
                "圖片比例",
                options=["1:1", "16:9", "9:16", "4:3", "3:4"],
                index=0,
                help="選擇生成圖片的寬高比"
            )
        
        # Add sticker style options when in LINE sticker mode
        sticker_style = None
        if line_sticker_mode:
            sticker_style = st.selectbox(
                "貼圖風格",
                options=["可愛卡通", "簡約線條", "表情豐富", "動物角色", "食物擬人化"],
                index=0,
                help="選擇 LINE 貼圖的視覺風格"
            )
        
        return {
            "prompt": prompt,
            "temperature": temperature,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "line_sticker_mode": line_sticker_mode,
            "sticker_style": sticker_style
        }

def render_image_uploader():
    uploaded_files = st.file_uploader(
        "上傳多張圖片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="支援 JPG、JPEG 和 PNG 格式，可同時上傳多張圖片"
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = []
        
        for uploaded_file in uploaded_files:
            # 為每個上傳的圖片生成唯一ID
            image_id = str(uuid.uuid4())
            
            # 儲存圖片到臨時目錄
            image = Image.open(uploaded_file)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, f"{image_id}.png")
            image.save(temp_path, format="PNG")
            
            # 添加到上傳的圖片列表
            st.session_state.uploaded_images.append({
                "id": image_id,
                "name": uploaded_file.name,
                "path": temp_path,
                "temp_dir": temp_dir
            })
    
    # 顯示所有上傳的圖片
    if st.session_state.uploaded_images:
        st.subheader(f"已上傳 {len(st.session_state.uploaded_images)} 張圖片")
        
        cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        for i, img_data in enumerate(st.session_state.uploaded_images):
            with cols[i % 3]:
                st.image(img_data["path"], caption=f"圖片 {i+1}: {img_data['name']}", use_container_width=True)

def process_images(params):
    if not st.session_state.uploaded_images:
        st.warning("請先上傳至少一張圖片！")
        return
    
    with st.spinner("🤖 AI 正在處理您的圖片..."):
        # 初始化進度條
        progress_bar = st.progress(0)
        total_steps = len(st.session_state.uploaded_images) * (1 + params["num_images"])
        completed_steps = 0
        
        # 清空舊的分析結果
        st.session_state.image_analyses = {}
        st.session_state.generated_images = {}
        
        # 處理每張上傳的圖片
        for img_data in st.session_state.uploaded_images:
            # 根據 LINE 貼圖模式調整提示詞
            effective_prompt = params["prompt"]
            if params["line_sticker_mode"]:
                sticker_style_prompts = {
                    "可愛卡通": "識別圖片中的主要人物、角色或物體，並詳細描述其特徵（如外觀、表情、姿勢等）。將其轉換為可愛卡通風格的LINE貼圖，保持相同的角色和表情特徵。",
                    "簡約線條": "識別圖片中的主要人物、角色或物體，並詳細描述其特徵（如外觀、表情、姿勢等）。將其轉換為簡約線條風格的LINE貼圖，保持相同的角色和表情特徵。",
                    "表情豐富": "識別圖片中的主要人物、角色或物體，並詳細描述其特徵（如外觀、表情、姿勢等）。將其轉換為表情誇張生動的LINE貼圖，保持相同的角色和表情特徵。",
                    "動物角色": "識別圖片中的主要人物、角色或物體，並詳細描述其特徵（如外觀、表情、姿勢等）。將其轉換為可愛動物風格的LINE貼圖，保持相同的角色和表情特徵。",
                    "食物擬人化": "識別圖片中的主要人物、角色或物體，並詳細描述其特徵（如外觀、表情、姿勢等）。將其轉換為食物擬人化風格的LINE貼圖，保持相同的角色和表情特徵。"
                }
                style_prompt = sticker_style_prompts.get(params["sticker_style"], "識別圖片中的主要人物、角色或物體，並詳細描述其特徵（如外觀、表情、姿勢等）。將其轉換為適合作為LINE貼圖的風格，保持相同的角色和表情特徵。")
                effective_prompt = f"{effective_prompt} {style_prompt}"
            
            # 生成文本分析
            text_analysis = st.session_state.ai_service.generate_text_from_image(
                img_data["path"],
                effective_prompt,
                params["temperature"]
            )
            
            if text_analysis:
                st.session_state.image_analyses[img_data["id"]] = text_analysis
                completed_steps += 1
                progress_bar.progress(completed_steps / total_steps)
                
                # 為 LINE 貼圖模式增強生成提示
                generation_prompt = text_analysis
                if params["line_sticker_mode"]:
                    # 從文本分析中提取關鍵元素
                    key_elements = text_analysis.split("。")[0] if "。" in text_analysis else text_analysis
                    
                    # 強化保持原始圖片元素的提示
                    sticker_enhancement = f"""
                    Create a LINE sticker featuring the exact same character/object as in the original image: {key_elements}
                    Style: {params["sticker_style"] if params["sticker_style"] else "cute cartoon"}
                    Must maintain the identity and key features of the original subject
                    Simple clear design with white or transparent background
                    Expressive and emotionally clear for messaging
                    Digital art style suitable for LINE stickers
                    """
                    
                    # 合併提示，確保保留原始圖片元素
                    generation_prompt = f"{key_elements} - {sticker_enhancement}"
                
                # 生成指定數量的新圖片
                generated_images = st.session_state.ai_service.generate_image_from_text(
                    generation_prompt,
                    num_outputs=params["num_images"],
                    num_inference_steps=params["num_inference_steps"],
                    aspect_ratio=params["aspect_ratio"]
                )
                
                if generated_images:
                    st.session_state.generated_images[img_data["id"]] = generated_images
                    completed_steps += params["num_images"]
                    progress_bar.progress(completed_steps / total_steps)
        
        # 完成
        progress_bar.progress(1.0)
        st.success("✨ 所有圖片處理完成！")

def display_results():
    if not st.session_state.image_analyses:
        return
    
    st.markdown("---")
    st.header("📊 處理結果")
    
    # 顯示每張圖片的分析結果和生成的圖片
    for i, img_data in enumerate(st.session_state.uploaded_images):
        image_id = img_data["id"]
        
        if image_id not in st.session_state.image_analyses:
            continue
        
        st.subheader(f"圖片 {i+1}: {img_data['name']}")
        
        # 創建三列布局：原圖、文本分析和生成的圖片
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img_data["path"], caption="原始圖片", use_container_width=True)
        
        with col2:
            st.markdown("##### 📝 生成的文本分析")
            st.write(st.session_state.image_analyses[image_id])
        
        # 顯示生成的圖片
        if image_id in st.session_state.generated_images:
            st.markdown("##### 🖼️ 生成的新圖像")
            
            # 計算每行顯示的圖片數量
            images_per_row = min(3, len(st.session_state.generated_images[image_id]))
            num_rows = (len(st.session_state.generated_images[image_id]) + images_per_row - 1) // images_per_row
            
            for row in range(num_rows):
                cols = st.columns(images_per_row)
                
                for col_idx in range(images_per_row):
                    img_idx = row * images_per_row + col_idx
                    
                    if img_idx < len(st.session_state.generated_images[image_id]):
                        with cols[col_idx]:
                            img_url = st.session_state.generated_images[image_id][img_idx]
                            st.image(
                                img_url,
                                caption=f"生成圖片 {img_idx+1}",
                                use_container_width=True
                            )
                            
                            # 添加下載按鈕
                            st.markdown(f"[下載圖片]('{img_url}')")

def cleanup_temp_files():
    # 清理臨時文件
    for img_data in st.session_state.uploaded_images:
        try:
            os.remove(img_data["path"])
            os.rmdir(img_data["temp_dir"])
        except Exception:
            pass

# ----------------- Main Application -----------------

def main():
    # 初始化 session state
    init_session_state()
    
    # 渲染頁面頭部
    render_header()
    
    # 渲染側邊欄配置
    params = render_sidebar()
    
    # 渲染圖片上傳區域
    render_image_uploader()
    
    # 處理按鈕
    if st.button("🚀 開始處理", type="primary"):
        process_images(params)
    
    # 顯示結果
    display_results()
    
    # 渲染頁腳
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>基於 Gemma-3-27b-it 和 Flux-Schnell 模型構建 | 使用 Replicate API</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"應用程序發生錯誤: {str(e)}")
    finally:
        # 清理臨時文件
        if 'uploaded_images' in st.session_state:
            cleanup_temp_files()
