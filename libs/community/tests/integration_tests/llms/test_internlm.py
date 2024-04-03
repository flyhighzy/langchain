""" Test InternLM"""

from langchain_community.llms.internlm import InternLM

def test_lmdeploy_call() -> None:
    """ test valid call to internlm deployed by lmdeploy"""
    llm = InternLM(
        endpoint_url="http://10.140.0.65:33434/v1/chat/completions",
        model="internlm2-chat-20b"
    )
    output = llm("hello",stop=["[UNUSED_TOKEN_145]"])
    print(output)
    assert isinstance(output, str)