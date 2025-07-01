
import streamlit as st
from llm_manager import get_hf_binary_sentiment, get_hf_emotion_sentiment
import io
import mimetypes
import re
from logger import get_logger
from time import time
from emotion_plot import plot_emotions_circle

logging = get_logger()
# --- Constants ---
MAX_FILE_SIZE_MB = 1
MAX_ITEM_LENGTH = 200
ITEM_SEPARATOR = "|||"


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time()
        logging.info(f"[{func.__name__}] | STARTED")
        result = func(*args, **kwargs)
        logging.info(f"[{func.__name__}] | ENDED | TIME: {time() - start}")
        return result
    return wrapper


@time_it
def sanitize_text(text):
    # Remove control characters, null bytes, escape sequences
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Optionally strip emojis (basic unicode range filtering)
    text = ''.join(c for c in text if c <= '\uFFFF')
    return text.strip()


@time_it
def handle_sentiment_text(st, hf_token):
    user_input = st.text_area("💬 Enter text for sentiment analysis", height=150)
    analyze_btn = st.button("Analyze Sentiment")

    if not analyze_btn:
        return
    if not hf_token:
        st.warning("Please enter the Hugging Face Token to proceed")
        return
    if not user_input.strip():
        st.warning("Please enter some text.")
        return

    try:
        sentiment_pipeline = get_hf_binary_sentiment(hf_token)
        results = sentiment_pipeline(user_input.strip())
        label = results[0]["label"]
        score = results[0]["score"]

        st.success("✅ Sentiment Result:")
        st.write(f"**Sentiment**: {label}")
        st.write(f"**Confidence**: {score:.4%}")
    except Exception as e:
        print(e)
        st.error(f"Error during sentiment analysis: {e}")


@time_it
def handle_sentiment_file_upload(st, hf_token):
    st.markdown("📄 **Upload a `.txt` file** (max **5 MB**, items separated by `|||`)")

    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    analyze_btn = st.button("Analyze Sentiment")

    if not analyze_btn:
        return
    if not hf_token:
        st.warning("Please enter the Hugging Face Token to proceed.")
        return
    if uploaded_file is None:
        st.warning("Please upload a file.")
        return
    
    uploaded_file.seek(0, io.SEEK_END)
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    # --- Size Check ---
    if size_mb > MAX_FILE_SIZE_MB:
        st.error("🚫 File too large. Please upload a file under 1 MB.")
        return
    
    # --- MIME Type Check ---
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type != "text/plain":
        st.error("🚫 Invalid MIME type. Only plain text files are allowed.")
        return
    
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        raw_items = content.split(ITEM_SEPARATOR)

        cleaned_items = []
        for idx, item in enumerate(raw_items):
            sanitized = sanitize_text(item)
            if not sanitized:
                continue  # skip empty
            if len(sanitized) > MAX_ITEM_LENGTH:
                sanitized = sanitized[:MAX_ITEM_LENGTH]
            cleaned_items.append((idx + 1, sanitized))

        if not cleaned_items:
            st.warning("No valid items found in the file.")
            return
        
        # Load model
        sentiment_pipeline = get_hf_binary_sentiment(hf_token)

        st.success(f"Analyzing {len(cleaned_items)} item(s)...")
        for idx, item_text in cleaned_items:
            result = sentiment_pipeline(item_text)[0]
            st.write(f"**Item {idx}** — *{item_text}*")
            st.write(f"→ Sentiment: `{result['label']}` (Confidence: `{result['score']:.2%}`)")
            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")


@time_it
def handle_emotion_text(st, hf_token):
    user_input = st.text_area("💬 Enter text for emotion analysis", height=150)
    analyze_btn = st.button("Analyze Emotion")

    if not analyze_btn:
        return
    if not hf_token:
        st.warning("Please enter the Hugging Face Token to proceed")
        return
    if not user_input.strip():
        st.warning("Please enter some text.")
        return

    try:
        sentiment_pipeline = get_hf_emotion_sentiment(hf_token)
        results = sentiment_pipeline(user_input.strip())[0]
        fig = plot_emotions_circle(results)
        st.plotly_chart(fig)
    except Exception as e:
        print(e)
        st.error(f"Error during sentiment analysis: {e}")


@time_it
def main():
        
    st.set_page_config(page_title="Sentiment Analysis Bot", page_icon="🧠")
    st.title("🧠 Sentiment Analysis Bot (HuggingFace)")

    st.markdown("Enter your Hugging Face token and a sentence to analyze its sentiment.")

    # User Inputs
    hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
    nav_option = st.radio(
        "📋 Choose Input Mode:",
        ["Select...", "📝 Single Text", "📁 Upload File", "🌀 Emotion Profile"],
        horizontal=True,
    )
    
    match nav_option:
        case "📝 Single Text":
            handle_sentiment_text(st, hf_token)
        case "📁 Upload File":
            handle_sentiment_file_upload(st, hf_token)
        case "🌀 Emotion Profile":
            handle_emotion_text(st, hf_token)
        case _:
            return
            

if __name__ == "__main__":
    main()