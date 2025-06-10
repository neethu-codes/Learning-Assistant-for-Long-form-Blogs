from uuid import uuid4
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os # Import os to check for Streamlit Cloud environment

CHUNK_SIZE = 1500
llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
    
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={"trust_remote_code":True,'device': 'cpu'}
        )
        
        # Determine persist_directory based on environment
        # Use /tmp/ on Streamlit Cloud for temporary storage, or a relative path locally
        is_streamlit_cloud = os.environ.get("STREAMLIT_SERVER_PORT") is not None
        if is_streamlit_cloud:
            # /tmp is a writable directory on Streamlit Cloud. Data here is ephemeral.
            persist_dir = "/tmp/vectorstore" 
            Path(persist_dir).mkdir(parents=True, exist_ok=True) # Ensure directory exists
        else:
            # For local runs, use a relative path within your project
            persist_dir = str(Path(__file__).parent / "resources" / "vectorstore")
            Path(persist_dir).mkdir(parents=True, exist_ok=True) # Ensure directory exists

        vector_store = Chroma(
            collection_name="real-estate",
            embedding_function=ef,
            persist_directory=persist_dir
        )


def process_urls(urls):
    global vector_store # Ensure vector_store is accessible


    # Check if vector_store was initialized by main.py or fallback
    if vector_store is None:
        yield "Components not initialized. Attempting initialization within process_urls."
        initialize_components()
        if vector_store is None: # If still None, something is wrong
            raise RuntimeError("Failed to initialize vector store components.")


    yield "Resetting vector store.."
    # Ensure vector_store is not None before trying to reset
    if vector_store:
        vector_store.reset_collection()
    else:
        raise RuntimeError("Vector store not initialized before reset attempt.")

    yield "Loading data.."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks.."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_overlap=200,
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database.."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    yield "Finished adding chunks to vector database."
   
def generate_answer(query):
    global llm, vector_store # Ensure llm and vector_store are accessible

    # Ensure components are initialized before use
    if llm is None or vector_store is None:
        # Fallback initialization if not done elsewhere
        initialize_components() 
        if llm is None or vector_store is None:
            raise RuntimeError("LLM or Vector DB is not initialized.")
    
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20}
    )
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")
    return result['answer'], sources