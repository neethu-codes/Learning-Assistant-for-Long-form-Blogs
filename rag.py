from uuid import uuid4
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

CHUNK_SIZE = 1500
llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9, max_tokens=500)
    
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={"trust_remote_code":True}
        )
        vector_store  = Chroma(
            collection_name = "real-estate",
            embedding_function = ef,
            persist_directory=str(Path(__file__).parent / "resources"/"vectorstore")
        )



def process_urls(urls):

    yield "Initializing components.."
    initialize_components()

    yield "Resetting vector store.."
    vector_store.reset_collection()

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
    vector_store.add_documents(docs,ids=uuids)

   
def generate_answer(query):
    
    if not vector_store:
        raise RuntimeError("Vector DB is not initialised")
    
    retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20}
    )
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=retriever)
    
    result = chain.invoke({"question":query}, return_only_outputs=True)
    sources = result.get("sources","")
    return result['answer'], sources


