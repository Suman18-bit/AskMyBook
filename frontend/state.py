"""
Session state initialization and helpers for the Streamlit app.

Streamlit reruns the entire script on every interaction, so anything
that needs to persist across reruns (the active workspace, whether the
"new workspace" input is open, etc.) has to live in st.session_state
rather than as a plain module-level variable.
"""
import streamlit as st

import api_client as api


def init_session_state():
    if "active_workspace_id" not in st.session_state:
        st.session_state.active_workspace_id = None

    if "workspaces" not in st.session_state:
        st.session_state.workspaces = []

    if "chat_messages" not in st.session_state:
        # Keyed by workspace_id so switching workspaces doesn't require
        # a fresh backend round-trip if we've already loaded that
        # workspace's history this session.
        st.session_state.chat_messages = {}

    if "documents" not in st.session_state:
        st.session_state.documents = {}

    if "show_new_workspace_input" not in st.session_state:
        st.session_state.show_new_workspace_input = False

    if "pending_delete_workspace_id" not in st.session_state:
        st.session_state.pending_delete_workspace_id = None

    if "backend_reachable" not in st.session_state:
        st.session_state.backend_reachable = True

    refresh_workspaces()


def refresh_workspaces():
    try:
        st.session_state.workspaces = api.list_workspaces()
        st.session_state.backend_reachable = True
    except Exception:
        st.session_state.backend_reachable = False
        return

    # If no workspace is active yet but workspaces exist, activate the
    # most recently created one so the user lands somewhere useful.
    if st.session_state.active_workspace_id is None and st.session_state.workspaces:
        set_active_workspace(st.session_state.workspaces[0]["id"])

    # If the active workspace was deleted elsewhere, clear it.
    active_ids = {w["id"] for w in st.session_state.workspaces}
    if st.session_state.active_workspace_id not in active_ids:
        st.session_state.active_workspace_id = None


def set_active_workspace(workspace_id: str):
    st.session_state.active_workspace_id = workspace_id
    load_workspace_data(workspace_id)


def load_workspace_data(workspace_id: str):
    """Loads chat history and document list for a workspace if not already cached."""
    if workspace_id not in st.session_state.chat_messages:
        try:
            st.session_state.chat_messages[workspace_id] = api.get_chat_history(workspace_id)
        except Exception:
            st.session_state.chat_messages[workspace_id] = []

    try:
        st.session_state.documents[workspace_id] = api.list_documents(workspace_id)
    except Exception:
        st.session_state.documents[workspace_id] = []


def get_active_workspace() -> dict | None:
    if not st.session_state.active_workspace_id:
        return None
    for ws in st.session_state.workspaces:
        if ws["id"] == st.session_state.active_workspace_id:
            return ws
    return None


def get_active_messages() -> list[dict]:
    ws_id = st.session_state.active_workspace_id
    if not ws_id:
        return []
    return st.session_state.chat_messages.get(ws_id, [])


def get_active_documents() -> list[dict]:
    ws_id = st.session_state.active_workspace_id
    if not ws_id:
        return []
    return st.session_state.documents.get(ws_id, [])