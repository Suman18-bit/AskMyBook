"""
AskMyBook RAG — Streamlit frontend.

This is a thin client over the FastAPI backend (main.py). It owns no
RAG logic itself — every retrieval and generation call goes over HTTP
to the backend, including the token stream consumed in stream_chat().
"""
import time
from pathlib import Path

import streamlit as st

import api_client as api
from state import (
    init_session_state, refresh_workspaces, set_active_workspace,
    get_active_workspace, get_active_messages, get_active_documents,
    load_workspace_data,
)

st.set_page_config(
    page_title="AskMyBook",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = Path(__file__).parent / "styles.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="dot"></span>AskMyBook</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.backend_reachable:
            st.markdown(
                '<div class="connection-error">⚠️ Can\'t reach the backend at '
                'localhost:8000. Start it with <code>uvicorn main:app --reload</code> '
                'from the backend/ folder.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Retry connection", use_container_width=True):
                refresh_workspaces()
                st.rerun()
            return

        render_workspace_manager()

        active_ws = get_active_workspace()
        if active_ws:
            st.markdown('<div class="sidebar-section-label">Documents</div>', unsafe_allow_html=True)
            render_document_manager(active_ws["id"])


def render_workspace_manager():
    st.markdown('<div class="sidebar-section-label">Workspaces</div>', unsafe_allow_html=True)

    active_id = st.session_state.active_workspace_id

    for ws in st.session_state.workspaces:
        is_active = ws["id"] == active_id
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"**{ws['name']}**" if is_active else ws["name"]
            if st.button(
                label,
                key=f"switch_{ws['id']}",
                use_container_width=True,
                help=f"{ws['document_count']} docs · {ws['message_count']} messages",
            ):
                set_active_workspace(ws["id"])
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{ws['id']}", help="Delete workspace"):
                st.session_state.pending_delete_workspace_id = ws["id"]
                st.rerun()

    # Confirmation flow for delete — a destructive action shouldn't be
    # one accidental click away.
    pending_id = st.session_state.pending_delete_workspace_id
    if pending_id:
        pending_ws = next((w for w in st.session_state.workspaces if w["id"] == pending_id), None)
        if pending_ws:
            st.warning(f"Delete **{pending_ws['name']}**? This removes all its documents and chat history.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm", key="confirm_delete", type="primary", use_container_width=True):
                    api.delete_workspace(pending_id)
                    st.session_state.chat_messages.pop(pending_id, None)
                    st.session_state.documents.pop(pending_id, None)
                    st.session_state.pending_delete_workspace_id = None
                    if st.session_state.active_workspace_id == pending_id:
                        st.session_state.active_workspace_id = None
                    refresh_workspaces()
                    st.rerun()
            with c2:
                if st.button("Cancel", key="cancel_delete", use_container_width=True):
                    st.session_state.pending_delete_workspace_id = None
                    st.rerun()

    st.divider()

    if st.session_state.show_new_workspace_input:
        new_name = st.text_input("Workspace name", key="new_ws_name", label_visibility="collapsed", placeholder="e.g. Research Papers")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create", key="confirm_create", type="primary", use_container_width=True):
                if new_name.strip():
                    created = api.create_workspace(new_name.strip())
                    refresh_workspaces()
                    set_active_workspace(created["id"])
                    st.session_state.show_new_workspace_input = False
                    st.rerun()
                else:
                    st.error("Give the workspace a name.")
        with c2:
            if st.button("Cancel", key="cancel_create", use_container_width=True):
                st.session_state.show_new_workspace_input = False
                st.rerun()
    else:
        if st.button("+ New workspace", use_container_width=True):
            st.session_state.show_new_workspace_input = True
            st.rerun()


def render_document_manager(workspace_id: str):
    docs = get_active_documents()

    if docs:
        for doc in docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<div class="doc-chip"><span class="doc-name">📄 {doc["filename"]}</span>'
                    f'<span>{doc["chunk_count"]} chunks</span></div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("✕", key=f"deldoc_{doc['filename']}", help="Remove document"):
                    with st.spinner("Removing..."):
                        api.delete_document(workspace_id, doc["filename"])
                    load_workspace_data(workspace_id)
                    st.rerun()
    else:
        st.caption("No documents yet. Upload below to get started.")

    uploaded = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "xlsx"],
        label_visibility="collapsed",
        key=f"uploader_{workspace_id}",
    )

    if uploaded is not None:
        already_uploaded = any(d["filename"] == uploaded.name for d in docs)
        if not already_uploaded:
            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    result = api.upload_document(workspace_id, uploaded.name, uploaded.getvalue())
                    st.toast(f"Added {result['chunks_created']} chunks from {uploaded.name}", icon="✅")
                    load_workspace_data(workspace_id)
                    st.rerun()
                except api.BackendError as e:
                    st.error(f"Upload failed: {e}")


def render_chat_area():
    active_ws = get_active_workspace()

    if not st.session_state.backend_reachable:
        return

    if not active_ws:
        st.markdown(
            '<div class="empty-state">'
            '<h3>No workspace selected</h3>'
            '<p>Create a workspace in the sidebar to start uploading documents and chatting.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"### {active_ws['name']}")

    docs = get_active_documents()
    if not docs:
        st.info("This workspace has no documents yet. Upload a file in the sidebar before asking questions — otherwise there's nothing to retrieve from.")

    messages = get_active_messages()
    chat_container = st.container()

    with chat_container:
        for msg in messages:
            role = msg["role"]
            with st.chat_message(role, avatar="🧑" if role == "user" else "✨"):
                st.markdown(
                    f'<div class="chat-bubble {role}">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    question = st.chat_input("Ask a question about your documents...")

    if question:
        ws_id = active_ws["id"]

        # Optimistically render the user's message immediately.
        st.session_state.chat_messages[ws_id].append({"role": "user", "content": question})
        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f'<div class="chat-bubble user">{question}</div>', unsafe_allow_html=True)

            # The signature retrieval-pulse element: shown while we wait
            # for the first token, representing the MMR search happening
            # server-side before generation starts.
            with st.chat_message("assistant", avatar="✨"):
                pulse_placeholder = st.empty()
                pulse_placeholder.markdown(
                    '<div class="retrieval-label">Retrieving context</div>'
                    '<div class="retrieval-pulse"></div>',
                    unsafe_allow_html=True,
                )

                response_placeholder = st.empty()
                accumulated = ""
                first_token_received = False

                try:
                    for event in api.stream_chat(ws_id, question):
                        if event["type"] == "token":
                            if not first_token_received:
                                pulse_placeholder.empty()
                                first_token_received = True
                            accumulated += event["content"]
                            response_placeholder.markdown(
                                f'<div class="chat-bubble assistant">{accumulated}▌</div>',
                                unsafe_allow_html=True,
                            )
                        elif event["type"] == "done":
                            response_placeholder.markdown(
                                f'<div class="chat-bubble assistant">{accumulated}</div>',
                                unsafe_allow_html=True,
                            )
                        elif event["type"] == "error":
                            pulse_placeholder.empty()
                            st.error(f"Generation error: {event['detail']}")

                except Exception as e:
                    pulse_placeholder.empty()
                    st.error(f"Connection lost while streaming: {e}")

        st.session_state.chat_messages[ws_id].append({"role": "assistant", "content": accumulated})
        st.rerun()

    if messages:
        if st.button("Clear chat history"):
            api.clear_chat_history(active_ws["id"])
            st.session_state.chat_messages[active_ws["id"]] = []
            st.rerun()


def main():
    init_session_state()
    load_css()
    render_sidebar()
    render_chat_area()


if __name__ == "__main__":
    main()