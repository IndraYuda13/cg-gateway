from typing import List, Optional, Any, Union, Dict
from pydantic import BaseModel, Field


class ImageUrlDetail(BaseModel):
    url: str
    detail: Optional[str] = "auto"
    model_config = {"extra": "allow"}


class FileUrlDetail(BaseModel):
    url: str
    name: Optional[str] = None
    mime_type: Optional[str] = None
    model_config = {"extra": "allow"}


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


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class ToolDefinition(BaseModel):
    type: str = "function"
    function: Optional[FunctionDefinition] = None
    model_config = {"extra": "allow"}


class ToolCallFunction(BaseModel):
    name: str
    arguments: str
    model_config = {"extra": "allow"}


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction
    model_config = {"extra": "allow"}


class MessageItem(BaseModel):
    role: str
    content: Optional[Union[str, List[Union[ContentPart, Dict[str, Any], str]]]] = ""
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    model_config = {"extra": "allow"}


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
    tools: Optional[List[Union[ToolDefinition, Dict[str, Any]]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True
    web_search: Optional[bool] = False
    model_config = {"extra": "allow"}


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
    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    model_config = {"extra": "allow"}


class ChoiceItem(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"
    model_config = {"extra": "allow"}


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_config = {"extra": "allow"}


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    session_id: Optional[str] = None
    choices: List[ChoiceItem]
    usage: UsageInfo
    model_config = {"extra": "allow"}


class ToolCallDeltaFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None
    model_config = {"extra": "allow"}


class ToolCallDelta(BaseModel):
    index: int
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[ToolCallDeltaFunction] = None
    model_config = {"extra": "allow"}


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCallDelta]] = None
    model_config = {"extra": "allow"}


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None
    model_config = {"extra": "allow"}


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    session_id: Optional[str] = None
    choices: List[ChatCompletionChunkChoice]
    model_config = {"extra": "allow"}


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "openai"
    root: str
    parent: Optional[str] = None
    model_config = {"extra": "allow"}


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelItem]
    model_config = {"extra": "allow"}
