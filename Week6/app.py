import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Sayfa ayarlarını en başta yap
st.set_page_config(page_title="Flan-T5 Chatbot", page_icon="🤖")

@st.cache_resource
def load_model():
    try:
        model_name = "google/flan-t5-base"
        
        # Progress bar göster
        progress_text = "Model yükleniyor... Lütfen bekleyin."
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # GPU varsa kullan
        if torch.cuda.is_available():
            model = model.to('cuda')
            
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

# Yükleme durumunu göster
with st.spinner('🔄 Model yükleniyor... İlk yüklemede 1-2 dakika sürebilir.'):
    tokenizer, model, error = load_model()

if error:
    st.error(f"Model yüklenemedi: {error}")
    st.stop()

def chatbot_response(prompt):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        if device == 'cuda':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=150,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# ---- Streamlit App UI ----
st.title("🤖 Flan-T5 Chatbot")
st.caption("Powered by Google Flan-T5")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Mesajınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor..."):
            response = chatbot_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
