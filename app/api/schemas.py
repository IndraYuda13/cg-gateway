from typing import List, Optional, Any, Union, Dict
from pydantic import BaseModel, Field


class ImageUrlDetail(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Union[ImageUrlDetail, Dict[str, Any], str]] = None
    model_config = {"extra": "allow"}


class MessageItem(BaseModel):
    role: str
    content: Union[str, List[Union[ContentPart, Dict[str, Any], str]]]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gpt-5-6-thinking"
    messages: List[MessageItem]
    stream: Optional[bool] = False
    thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking_effort: Optional[str] = None
    session_id: Optional[str] = None
    new_session: Optional[bool] = False
    user: Optional[str] = None
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


class SimpleChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-5-6-thinking"
    thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking_effort: Optional[str] = None
    stream: Optional[bool] = False
    new_session: Optional[bool] = False
    user: Optional[str] = None
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str
    reasoning_content: Optional[str] = None


class ChoiceItem(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    session_id: Optional[str] = None
    choices: List[ChoiceItem]
    usage: UsageInfo


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "openai"
    root: str
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelItem]
