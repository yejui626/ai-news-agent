# llm.py
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json

def openai_llm(temperature=0):
    """Return an OpenAI-based LangChain Chat Model."""
    return ChatOpenAI(
        model="gpt-4o-mini",  # or gpt-4-turbo
        temperature=temperature
    )

def generate_string(llm, prompt_str: str, input_vars: dict, show_prompt=False, system_prompt_only=True):
    """
    Generate string output from LLM given a structured prompt.
    """
    messages = []
    if system_prompt_only:
        messages.append(SystemMessage(content=prompt_str))
    else:
        messages.append(HumanMessage(content=prompt_str))

    if show_prompt:
        print("\n===== PROMPT SENT TO LLM =====\n")
        print(prompt_str)
        print("\n===============================\n")

    response = llm.invoke(messages)
    return response

def generate_json(llm, prompt_str: str, schema: dict, show_prompt=False):
    """
    Generate JSON output from LLM and parse it into dict.
    """
    response = generate_string(llm, prompt_str, {}, show_prompt)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"raw_response": response.content}
    return parsed
