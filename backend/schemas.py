from typing import List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    threshold: Optional[float] = None
    model: Optional[str] = None


class RouteRequest(BaseModel):
    question: str
    threshold: Optional[float] = None
