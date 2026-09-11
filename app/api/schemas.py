from typing import List, Optional, Any, Union, Dict
from pydantic import BaseModel, Field


class ImageUrlDetail(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class FileUrlDetail(BaseModel):
    url: str
    name: Optional[str] = None
    mime_type: Optional[str] = None


class ContentPart(BaseModel):
    type: str  # "text", "image_url", "file_url", "file", "image"
    text: Optional[str] = None
    image_url: Optional[Union[ImageUrlDetail, Dict[str, Any], str]] = None
    file_url: Optional[Union[FileUrlDetail, Dict[str, Any], str]] = None
    file: Optional[Union[Dict[str, Any], str]] = None
    image: Optional[Union[Dict[str, Any], str]] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None
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
    history_and_training_disabled: Optional[bool] = True
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
    history_and_training_disabled: Optional[bool] = True
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
