from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
from queue import Queue, Empty
import threading
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs):
        # Signal end of stream with None
        self.queue.put(None)

    def on_llm_error(self, error: Exception, **kwargs):
        self.queue.put(f"[ERROR] {str(error)}")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

chat_histories = {}

def build_rag_chain(llm_source="openai", callbacks=None):
    loader = PyPDFLoader("data/nova_przestrzen.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    if llm_source == "openai":
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            streaming=True,
            callbacks=callbacks,
            temperature=0
        )
    elif llm_source == "ollama":
        llm = OllamaLLM(model="mistral", temperature=0)
    else:
        raise ValueError("Unknown LLM source")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

def handle_chat_stream(data, llm_source="openai"):
    user_message = data.get("message")
    session_id = data.get("session_id")

    if not user_message and not chat_histories.get(session_id):
        return Response("data: Hello! How can I help you today?\n\ndata: [END]\n\n", mimetype="text/event-stream")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]
    streaming_handler = StreamingCallbackHandler()

    def generate():
        try:
            rag_chain = build_rag_chain(llm_source=llm_source, callbacks=[streaming_handler])
            inputs = {"question": user_message, "chat_history": chat_history}
            output = ""

            for chunk in rag_chain.stream(inputs):
                token = chunk["answer"]
                output += token
                yield f"data: {token}\n\n"

            chat_history.append((user_message, output))
            yield "data: [END]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/new-session', methods=['GET', 'OPTIONS'])
def new_session():
    session_id = str(uuid.uuid4())
    chat_histories[session_id] = []
    welcome_message = "Hello! How can I help you today?"
    chat_histories[session_id].append(("system", welcome_message))
    return jsonify({"session_id": session_id, "welcome_message": welcome_message})

@app.route('/chat-stream', methods=['GET', 'OPTIONS'])
def chat_stream():
    user_message = request.args.get("message")
    session_id = request.args.get("session_id")
    data = {"message": user_message, "session_id": session_id}
    return handle_chat_stream(data, llm_source="openai")

@app.route('/chat-stream-local', methods=['GET'])
def chat_stream_local():
    user_message = request.args.get("message")
    session_id = request.args.get("session_id")
    data = {"message": user_message, "session_id": session_id}
    return handle_chat_stream(data, llm_source="ollama")

if __name__ == '__main__':
    app.run(debug=True)