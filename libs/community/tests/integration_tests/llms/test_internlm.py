""" Test InternLM"""

from langchain_community.llms.internlm import InternLM

def test_lmdeploy_call() -> None:
    """ test valid call to internlm deployed by lmdeploy"""
    llm = InternLM(
        endpoint_url="http://127.0.0.1:23333/v1/chat/completions",
        model="internlm2-chat-20b"
    )
    output = llm("hello",stop=["[UNUSED_TOKEN_145]"])
    print(output)
    assert isinstance(output, str)