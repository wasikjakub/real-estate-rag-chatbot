from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM


def build_rag_chain(config, callbacks=None, llm_source: str = "openai"):
    """Construct the retrieval-augmented generation pipeline.

    Args:
        config: AppConfig with env settings.
        callbacks: Optional callbacks for streaming.
        llm_source: "openai" or "ollama" to choose backend.
    """
    loader = PyPDFLoader(config.pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    if llm_source == "openai":
        llm = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model=config.openai_model,
            streaming=True,
            callbacks=callbacks,
            temperature=0,
        )
    elif llm_source == "ollama":
        llm = OllamaLLM(model="mistral", temperature=0)
    else:
        raise ValueError("Unknown LLM source")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
    )


