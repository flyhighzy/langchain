from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import Field, root_validator

from langchain_community.llms.utils import enforce_stop_tokens
import requests

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {"role": "function", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


class Backend():
    """
        basic backend class
    """
    def __init__(self, endpoint, prefix_messages: List[BaseMessage], timeout: int, **model_kwargs) -> None:
        self.endpoint = endpoint
        self.prefix_messages = prefix_messages
        self._kwargs = model_kwargs
        self.timeout = timeout

    def call(self, prompt: str, stop: Optional[List[str]] = None): # TODO response type
        raise NotImplementedError

    async def asyncCall(self, prompt: str):  # TODO response type
        raise NotImplementedError


class LMDeployBackend(Backend):
    """
        use lmdeploy to connect to InternLM inference api
    """
    def call(self, prompt: str, stop: Optional[List[str]] = None):
        headers = {
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=self._get_payload(prompt, stop),
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                parsed_json = json.loads(response.text)
                logger.info(f"get response: {parsed_json}")
                print(parsed_json)
                return parsed_json["choices"][0]["message"]["content"]
            else:
                logger.info(f"response: {response.content}")
                response.raise_for_status()
        except Exception as e:
            raise ValueError(f"An error has occurred: {e}")


    def _get_payload(self, prompt: str, stop: Optional[List[str]] = None) -> dict:
        params = self._invocation_params()
        messages = self.prefix_messages + [HumanMessage(content=prompt)]
        params.update(
            {
                "messages": [_convert_message_to_dict(m) for m in messages],
            }
        )
        if stop is not None:
            params["stop"] = stop.pop()
        return params

    def _invocation_params(self) -> dict:
        body = self._kwargs
        body["model"] = self._kwargs.get("model", "internlm2")  # required param for lmdeploy api
        return body

class InternLMApiType(str, Enum):
    """
    We'll support more API type in the future, like vLLM and public API.
    currently we only support to use lmdeploy to deploy InternLM on local GPU or cloud.
    """
    lmdeploy = "lmdeploy"

backend_mapping = {
    InternLMApiType.lmdeploy: LMDeployBackend
}

class InternLM(LLM):
    """InternLM LLM service."""

    endpoint_url: str = ""
    """InternLM endpoint"""

    endpoint_api_type: InternLMApiType = InternLMApiType.lmdeploy
    """Type of the endpoint being consumed. Possible values are `lmdeploy` for lmdeploy deployed models, and will have more options """
    
    model = "internlm2"
    """model name of InternLm, default is `internlm2`"""

    temperature: float = 0.7
    """What sampling temperature to use."""
    top_p: float = 0.85
    """What probability mass to use."""

    max_tokens: int = 512
    """response max tokens"""
    streaming: bool = False
    """Whether to stream the results or not."""
    
    repetition_penalty: float = 1.0
    """ The parameter for repetition penalty, 1.0 means no penalty"""

    prefix_messages: List[BaseMessage] = Field(default_factory=list)

    timeout: int = 120

    model_kwargs: Optional[dict] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "internlm"

    @property
    def _invocation_params(self) -> dict:
        """Get the parameters used to invoke the model."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "repetition_penalty": self.repetition_penalty
        }
        return {**params, **(self.model_kwargs or {})}
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if values["endpoint_api_type"] is None or values["endpoint_api_type"] == "":
            values["endpoint_api_type"] = InternLMApiType.lmdeploy
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
            get real backend and call api
        """
        real_backend_cls = backend_mapping.get(self.endpoint_api_type)
        if not real_backend_cls:
            logger.error(f"not valid endpoint_api_type: {self.endpoint_api_type}")
            return None
        real_backend = real_backend_cls(self.endpoint_url, self.prefix_messages, timeout=self.timeout, **self._invocation_params)
        return real_backend.call(prompt=prompt, stop=stop)
