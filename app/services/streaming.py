import json
from queue import Queue

from flask import Response
from langchain.callbacks.base import BaseCallbackHandler


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, error: Exception, **kwargs):
        self.queue.put(f"[ERROR] {str(error)}")


def stream_chat_response(rag_chain, user_message: str, chat_history: list):
    """Return a Flask streaming Response for a chat turn using SSE."""

    def generate():
        try:
            inputs = {"question": user_message, "chat_history": chat_history}
            output = ""

            for chunk in rag_chain.stream(inputs):
                token = chunk["answer"]
                print(token, end="", flush=True)
                output += token

                # Encode safely for SSE, handling newlines and quotes
                safe_token = json.dumps(token)[1:-1]
                yield f"data: {safe_token}\n\n"

            chat_history.append((user_message, output))
            yield "data: [END]\n\n"

        except Exception as e:
            print(f"Error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


