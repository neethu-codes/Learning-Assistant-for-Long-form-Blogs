import streamlit as st
import os
import sys

# --- Conditional Check for Streamlit Cloud ---
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SERVER_PORT") is not None

# --- SQLite3 Fix (ONLY on Streamlit Cloud) ---
if IS_STREAMLIT_CLOUD:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("Running on Streamlit Cloud: Applied pysqlite3 fix.")
    except ImportError:
        st.error("Error: pysqlite3-binary not found on Streamlit Cloud. Please ensure it's in your requirements.txt.")
        st.stop() 
else:
    print("Running locally: pysqlite3 fix skipped.")

# --- API Key Loading ---
# 1. On Streamlit Cloud: Load from st.secrets
if IS_STREAMLIT_CLOUD:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        print("Running on Streamlit Cloud: Groq API key loaded from st.secrets.")
    else:
        st.error("Groq API key not found in Streamlit secrets. Please add 'GROQ_API_KEY' to your app's secrets in the Cloud dashboard.")
        st.stop()
# 2. Locally: Load from .env file (if it exists)
else:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Running locally: .env file attempted to load.")
    except ImportError:
        print("Running locally: 'python-dotenv' not found. Ensure API keys are set in your environment manually.")
    except Exception as e:
        print(f"Running locally: Error loading .env file: {e}")

    # After attempting to load from .env (or if not using .env), check if key is now in os.environ
    if "GROQ_API_KEY" not in os.environ:
        st.error("Groq API key not found in your local environment. "
                   "Please set the GROQ_API_KEY environment variable (e.g., in your .env file or terminal).")
        st.stop()
    else:
        print("Running locally: Groq API key found in OS environment.")


from rag import process_urls, generate_answer



st.title("Personal Learning Assistant Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
url4 = st.sidebar.text_input("URL 4")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")


if process_url_button:
    urls = [url for url in (url1, url2, url3, url4) if url!='']
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid url")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must process urls first")
