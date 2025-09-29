from flask import Blueprint, jsonify, render_template, request, current_app

from ..config import AppConfig
from ..services.rag import build_rag_chain
from ..services.streaming import StreamingCallbackHandler, stream_chat_response
from ..state import chat_histories


chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/")
def home():
    return render_template("index.html")


@chat_bp.route("/new-session", methods=["GET", "OPTIONS"])
def new_session():
    config: AppConfig = current_app.config["APP_CONFIG"]

    import uuid

    session_id = str(uuid.uuid4())
    system_prompt = (
        "You are a helpful and friendly virtual assistant for Nova Przestrzeń. "
        "Answer questions clearly and politely, always referencing our offer. "
        "Use a professional but approachable tone."
        "Don't say that you have access to some document, just answer like you know all the info."
    )

    chat_histories[session_id] = [("system", system_prompt)]
    welcome_message = (
        "Hello! I am the Nova Przestrzeń virtual assistant. I am here to answer all your questions regarding our offer."
    )
    chat_histories[session_id].append(("system", welcome_message))
    return jsonify({"session_id": session_id, "welcome_message": welcome_message})


@chat_bp.route("/chat-stream", methods=["GET", "OPTIONS"])
def chat_stream():
    config: AppConfig = current_app.config["APP_CONFIG"]
    user_message = request.args.get("message")
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]
    streaming_handler = StreamingCallbackHandler()
    rag_chain = build_rag_chain(config=config, callbacks=[streaming_handler], llm_source="openai")
    return stream_chat_response(rag_chain, user_message, chat_history)


@chat_bp.route("/chat-stream-local", methods=["GET"]) 
def chat_stream_local():
    config: AppConfig = current_app.config["APP_CONFIG"]
    user_message = request.args.get("message")
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]
    streaming_handler = StreamingCallbackHandler()
    rag_chain = build_rag_chain(config=config, callbacks=[streaming_handler], llm_source="ollama")
    return stream_chat_response(rag_chain, user_message, chat_history)


