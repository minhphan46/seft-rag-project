import os
import re
import html
import threading
from typing import List, Dict

import streamlit as st
import torch
from transformers import TextIteratorStreamer
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    # Fallback for environments where top-level lazy import fails.
    import transformers

    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer


MODEL_DIR = r"C:\Users\minhp\Desktop\development\seft-rag-project\selfrag_llama2_7b"
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "_chat_ui_offload")
os.makedirs(OFFLOAD_DIR, exist_ok=True)

SUGGESTED_QUESTIONS = [
    "Can you tell me the difference between llamas and alpacas?",
    "Leave odd one out: twitter, instagram, whatsapp.",
    "What is overfitting in machine learning?",
    "Explain retrieval-augmented generation in simple terms.",
]


def build_prompt(question: str) -> str:
    return f"### Instruction:\n{question.strip()}\n\n### Response:\n"


def clean_final_answer(text: str) -> str:
    cleaned = text.strip().replace("<s>", "").replace("</s>", "").strip()
    if "### Response:" in cleaned:
        cleaned = cleaned.split("### Response:", 1)[1].strip()

    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    cleaned = re.sub(r"</?paragraph>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return (
            "Model is requesting retrieval context (`<paragraph>`), "
            "so there is no grounded final answer yet."
        )
    return cleaned


@st.cache_resource(show_spinner="Loading tokenizer and model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kw = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        load_kw["device_map"] = "auto"
        load_kw["offload_folder"] = OFFLOAD_DIR
    else:
        load_kw["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **load_kw)
    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.generation_config.max_length = None
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    return tokenizer, model


def generate_answer_stream(
    question: str,
    max_new_tokens: int = 128,
    on_stream_text=None,
) -> Dict[str, str]:
    tokenizer, model = load_model()
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    embed_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    generation_error = {"error": None}

    def _run_generation():
        try:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                )
        except Exception as exc:
            generation_error["error"] = exc

    thread = threading.Thread(target=_run_generation, daemon=True)
    thread.start()

    raw_chunks: List[str] = []
    for token_text in streamer:
        raw_chunks.append(token_text)
        if on_stream_text is not None:
            on_stream_text("".join(raw_chunks))

    thread.join()
    if generation_error["error"] is not None:
        raise generation_error["error"]

    raw_trace = "".join(raw_chunks).strip()
    final_answer = clean_final_answer(raw_trace)
    return {"trace": raw_trace, "answer": final_answer}


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "draft_question" not in st.session_state:
        st.session_state.draft_question = ""


def trace_html(trace_text: str) -> str:
    return (
        "<div style='font-size: 0.82rem; line-height: 1.35; "
        "white-space: pre-wrap;'>"
        f"{html.escape(trace_text)}"
        "</div>"
    )


def render_trace_text(trace_text: str):
    st.markdown(trace_html(trace_text), unsafe_allow_html=True)


def send_question(question: str, max_new_tokens: int = 128, render_container=None):
    question = question.strip()
    if not question:
        st.warning("Please type a question first.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    target = render_container if render_container is not None else st
    with target:
        with st.chat_message("user"):
            st.markdown(question)

        trace = ""
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            trace_placeholder = st.empty()
            answer_placeholder = st.empty()

            status_state = {"tick": 0}

            def update_loading_status():
                dots = "." * ((status_state["tick"] % 3) + 1)
                status_placeholder.markdown(
                    (
                        "<div style='display:flex; align-items:center; gap:10px; "
                        "padding:12px 14px; margin-bottom:10px;"
                        "border-radius:10px; background:#e9eff8; color:#3b5978;'>"
                        "<span style='width:14px; height:14px; border:2px solid #8ea6c2; "
                        "border-top-color:#3b82f6; border-radius:50%; display:inline-block; "
                        "animation: spin 0.9s linear infinite;'></span>"
                        f"<span>Generating response{dots}</span>"
                        "</div>"
                        "<style>"
                        "@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }"
                        "</style>"
                    ),
                    unsafe_allow_html=True,
                )
                status_state["tick"] += 1

            update_loading_status()
            with trace_placeholder.container():
                with st.expander("Reasoning / generation trace (live)", expanded=True):
                    live_trace_placeholder = st.empty()
                    try:
                        result = generate_answer_stream(
                            question,
                            max_new_tokens=max_new_tokens,
                            on_stream_text=lambda text: (
                                update_loading_status(),
                                live_trace_placeholder.markdown(
                                    trace_html(text),
                                    unsafe_allow_html=True,
                                ),
                            ),
                        )
                        answer = result["answer"]
                        trace = result["trace"]
                    except Exception as exc:
                        answer = f"Error while generating answer: {exc}"
                        trace = ""

            status_placeholder.empty()
            trace_placeholder.empty()
            if trace:
                with st.expander("Reasoning trace", expanded=False):
                    render_trace_text(trace)
            answer_placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
    st.session_state.draft_question = ""


def main():
    st.set_page_config(page_title="Self-RAG Chat UI", page_icon="💬", layout="wide")
    st.title("💬 Self-RAG Chatbot")
    st.caption("ChatGPT-style UI in Python for your local Self-RAG model.")

    init_state()
    left_col, right_col = st.columns([3, 1], gap="large")

    with right_col:
        st.subheader("Suggested Questions")
        selected = st.radio("Choose a suggestion", SUGGESTED_QUESTIONS, index=0)
        if st.button("Use selected suggestion", use_container_width=True):
            st.session_state.draft_question = selected

        if st.button("Send selected suggestion now", use_container_width=True):
            send_question(
                selected,
                max_new_tokens=st.session_state.get("max_new_tokens", 128),
            )
            st.rerun()

        st.divider()
        max_new_tokens = st.slider("Max new tokens", min_value=32, max_value=256, value=128, step=16)
        st.session_state.max_new_tokens = max_new_tokens

    with left_col:
        st.subheader("Conversation")
        info_placeholder = st.empty()
        if not st.session_state.messages or len(st.session_state.messages) == 0:
            info_placeholder.info("Start by selecting a suggested question or typing your own.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                trace = msg.get("trace", "")
                if msg["role"] == "assistant" and trace:
                    with st.expander("Reasoning trace", expanded=False):
                        render_trace_text(trace)

        live_response_container = st.container()

        with st.form("chat_form", clear_on_submit=False):
            question = st.text_area(
                "Enter your question",
                value=st.session_state.draft_question,
                placeholder="Type your question here...",
                height=110,
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted:
            info_placeholder.empty()
            st.session_state.draft_question = question
            send_question(
                st.session_state.draft_question,
                max_new_tokens=st.session_state.get("max_new_tokens", 128),
                render_container=live_response_container,
            )
            st.rerun()


if __name__ == "__main__":
    main()
