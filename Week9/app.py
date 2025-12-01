import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
import torch

# Page Configuration
st.set_page_config(page_title="AI Creative Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Creative Assistant")
st.caption("Powered by Google Flan-T5 & Stable Diffusion")

# Sidebar for Mode Selection
# Sidebar'ı varsayılan olarak açık tutmaya çalışalım ama kullanıcı manuel açmalı
mode = st.sidebar.selectbox("Select Mode", ["Chat Mode", "Art Mode"])
st.sidebar.markdown("---")
st.sidebar.write("Switch between chatting and image generation.")

# --- MODEL LOADING FUNCTIONS ---

@st.cache_resource
def load_chat_model():
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            model = model.to('cuda')
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_image_model():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        return pipeline, None
    except Exception as e:
        return None, str(e)

# --- CHAT MODE LOGIC ---
if mode == "Chat Mode":
    st.header("💬 Chat Mode")

    # Load Chat Model
    with st.spinner('Loading Chat Model...'):
        tokenizer, chat_model, chat_error = load_chat_model()

    if chat_error:
        st.error(f"Error loading chat model: {chat_error}")
        st.stop()

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

                    if device == 'cuda':
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():

                        outputs = chat_model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=150,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.9,
                            no_repeat_ngram_size=2,
                            early_stopping=True
                        )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# --- ART MODE LOGIC ---
elif mode == "Art Mode":
    st.header("🎨 Art Mode")
    st.info("Creating images requires heavy GPU usage. Please be patient.")

    # Load Image Model
    with st.spinner('Loading Image Model...'):
        image_pipeline, img_error = load_image_model()

    if img_error:
        st.error(f"Error loading image model: {img_error}")
        st.stop()

    prompt = st.text_input("Describe the image you want to generate:", placeholder="e.g. A futuristic city with flying cars, cyberpunk style")
    generate_btn = st.button("Generate Image")

    if generate_btn and prompt:
        with st.spinner("Generating masterpiece... (This may take a moment)"):
            try:
                image = image_pipeline(prompt).images[0]
                st.image(image, caption=prompt, use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
