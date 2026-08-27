"""
Pydantic models for request/response validation across the API.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class WorkspaceInfo(BaseModel):
    id: str
    name: str
    created_at: datetime
    document_count: int
    message_count: int


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceInfo]


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    workspace_id: str
    messages: list[ChatMessage]


class QueryRequest(BaseModel):
    workspace_id: str
    question: str = Field(..., min_length=1)


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    uploaded_at: datetime
    file_type: str


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str


class DeleteDocumentRequest(BaseModel):
    workspace_id: str
    filename: str