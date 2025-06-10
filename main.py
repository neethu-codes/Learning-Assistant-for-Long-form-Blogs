import sys
import os
import streamlit as st 


# 1. Determine if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SERVER_PORT") is not None
if IS_STREAMLIT_CLOUD:
    print("Detected running on Streamlit Cloud.")
else:
    print("Detected running locally.")

# 2. SQLite3 Fix (Conditional for Streamlit Cloud)
if IS_STREAMLIT_CLOUD:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("Streamlit Cloud: Successfully patched sqlite3 to use pysqlite3-binary.")
    except ImportError:
        st.error("FATAL ERROR: 'pysqlite3-binary' not found. "
                 "Please ensure 'pysqlite3-binary' is in your requirements.txt. "
                 "Check your Streamlit Cloud logs for installation errors.")
        st.stop()
    except Exception as e:
        st.error(f"FATAL ERROR during sqlite3 patch: {e}. Please report this.")
        st.stop()
else:
    print("Local run: Skipping sqlite3 patch.")

# 3. API Key Loading (Conditional for Local vs. Cloud)
# This ensures your GROQ_API_KEY is available in os.environ for your app.
if IS_STREAMLIT_CLOUD:
    # On Streamlit Cloud, load from st.secrets
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        print("Streamlit Cloud: Groq API key loaded from st.secrets.")
    else:
        st.error("Groq API key not found in Streamlit secrets. "
                 "Please add 'GROQ_API_KEY' to your app's secrets in the Streamlit Cloud dashboard.")
        st.stop()
else:
    # Locally, attempt to load from .env file using python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Local run: Attempted to load .env file.")
    except ImportError:
        print("Local run: 'python-dotenv' not found. Ensure API keys are set in your environment manually.")
    except Exception as e:
        print(f"Local run: Error loading .env file: {e}")

    # After attempting to load, verify the key is in os.environ
    if "GROQ_API_KEY" not in os.environ:
        st.error("Groq API key not found in your local environment. "
                   "Please set the GROQ_API_KEY environment variable (e.g., in your .env file or terminal) "
                   "before running your app locally.")
        st.stop()
    else:
        print("Local run: Groq API key found in OS environment.")


from rag import process_urls, generate_answer, initialize_components

initialize_components()

st.set_page_config(layout="wide")
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
        placeholder.text(f"Error: {e}. You must process URLs first.")