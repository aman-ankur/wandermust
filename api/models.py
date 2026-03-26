"""Pydantic models for the Discovery v2 conversation API.

ConversationTurn is the core response type — every API response returns one.
The LLM generates it. The UI renders it.
"""
from typing import List, Optional

from pydantic import BaseModel


class Option(BaseModel):
    id: str
    label: str
    insight: str
    emoji: Optional[str] = None


class DestinationHint(BaseModel):
    name: str
    hook: str
    match_reason: str


class ConversationTurn(BaseModel):
    phase: str  # "profile" | "discovery" | "narrowing" | "reveal"
    reaction: Optional[str] = None
    question: str
    options: List[Option]
    multi_select: bool = False
    can_free_text: bool = True
    destination_hints: Optional[List[DestinationHint]] = None
    thinking: Optional[str] = None
    phase_complete: bool = False
    topic: Optional[str] = None


class DiscoveryStartRequest(BaseModel):
    user_id: str = "default"


class DiscoveryRespondRequest(BaseModel):
    session_id: str
    answer: str
    option_ids: Optional[List[str]] = None


class DiscoverySelectRequest(BaseModel):
    session_id: str
    destination: str
