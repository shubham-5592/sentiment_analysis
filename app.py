
import streamlit as st
from llm_manager import get_hf_sentiment_chain
import io
import mimetypes
import re

# --- Constants ---
MAX_FILE_SIZE_MB = 1
MAX_ITEM_LENGTH = 200
ITEM_SEPARATOR = "|||"


def sanitize_text(text):
    # Remove control characters, null bytes, escape sequences
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Optionally strip emojis (basic unicode range filtering)
    text = ''.join(c for c in text if c <= '\uFFFF')
    return text.strip()


def handle_text_input(user_input, st, hf_token):
    try:
        sentiment_pipeline = get_hf_sentiment_chain(hf_token)
        results = sentiment_pipeline(user_input.strip())
        label = results[0]["label"]
        score = results[0]["score"]

        st.success("✅ Sentiment Result:")
        st.write(f"**Sentiment**: {label}")
        st.write(f"**Confidence**: {score:.4%}")
    except Exception as e:
        print(e)
        st.error(f"Error during sentiment analysis: {e}")


def handle_file_upload(uploaded_file, st, hf_token):

    uploaded_file.seek(0, io.SEEK_END)
    size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    # --- Size Check ---
    if size_mb > MAX_FILE_SIZE_MB:
        st.error("🚫 File too large. Please upload a file under 5 MB.")
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
        sentiment_pipeline = get_hf_sentiment_chain(hf_token)

        st.success(f"Analyzing {len(cleaned_items)} item(s)...")
        for idx, item_text in cleaned_items:
            result = sentiment_pipeline(item_text)[0]
            st.write(f"**Item {idx}** — *{item_text}*")
            st.write(f"→ Sentiment: `{result['label']}` (Confidence: `{result['score']:.2%}`)")
            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")


def main():
        
    st.set_page_config(page_title="Sentiment Analysis Bot", page_icon="🧠")
    st.title("🧠 Sentiment Analysis Bot (HuggingFace)")

    st.markdown("Enter your Hugging Face token and a sentence to analyze its sentiment.")

    # User Inputs
    hf_token = st.text_input("🔐 Hugging Face API Token", type="password")
    user_input = st.text_area("💬 Enter text for sentiment analysis", height=150)
    uploaded_file = st.file_uploader("📄 Upload a `.txt` file (Max 1 MB). each input must be separated by `|||`", type=["txt"])
    analyze_btn = st.button("Analyze Sentiment")

    # Run analysis
    if analyze_btn:
        if not hf_token:
            st.warning("Please enter the Hugging Face Token to proceed")
        elif not user_input.strip() and uploaded_file is None:
            st.warning("Please enter either some text or upload a file.")
        elif user_input.strip() not in (None, ""):
            handle_text_input(user_input, st, hf_token)
        else:
            handle_file_upload(uploaded_file, st, hf_token)
            

if __name__ == "__main__":
    main()