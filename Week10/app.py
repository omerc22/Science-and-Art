import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(page_title="AI Creative Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Creative Assistant")
st.caption("Powered by Google Flan-T5 & Stable Diffusion")

mode = st.sidebar.selectbox("Select Mode", ["Chat Mode", "Art Mode"])
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Creativity Parameters")

temperature = st.sidebar.slider("Temperature (Chat)", 0.1, 2.0, 0.7, 0.1)
max_length = st.sidebar.slider("Max Length (Chat)", 50, 512, 150, 10)
top_p = st.sidebar.slider("Top P (Chat)", 0.1, 1.0, 0.9, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Art Settings")
guidance_scale = st.sidebar.slider("Guidance Scale (Art)", 1.0, 20.0, 7.5, 0.5)
num_steps = st.sidebar.slider("Inference Steps (Art)", 10, 100, 30, 5)

st.sidebar.markdown("---")
st.sidebar.write(f"Current Mode: {mode}")

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

if mode == "Chat Mode":
    st.header("💬 Chat Mode")
    with st.spinner('Loading Chat Model...'):
        tokenizer, chat_model, chat_error = load_chat_model()

    if chat_error:
        st.error(f"Error: {chat_error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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

                    outputs = chat_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        no_repeat_ngram_size=2
                    )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

elif mode == "Art Mode":
    st.header("🎨 Art Mode")
    with st.spinner('Loading Image Model...'):
        image_pipeline, img_error = load_image_model()

    if img_error:
        st.error(f"Error: {img_error}")
        st.stop()

    prompt = st.text_input("Describe the image:", placeholder="e.g. A futuristic city, cyberpunk style")
    generate_btn = st.button("Generate Image")

    if generate_btn and prompt:
        with st.spinner("Generating..."):
            try:
                image = image_pipeline(
                    prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_steps
                ).images[0]
                st.image(image, caption=f"Prompt: {prompt}", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
