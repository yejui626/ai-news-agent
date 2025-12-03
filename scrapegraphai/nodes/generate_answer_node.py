"""
GenerateAnswerNode Module
"""

import json
import time
from typing import List, Optional

# ✅ NEW IMPORTS FOR AGENTIC FLOW
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from requests.exceptions import Timeout
from tqdm import tqdm

from ..prompts import (
    TEMPLATE_CHUNKS,
    TEMPLATE_CHUNKS_MD,
    TEMPLATE_MERGE,
    TEMPLATE_MERGE_MD,
    TEMPLATE_NO_CHUNKS,
    TEMPLATE_NO_CHUNKS_MD,
)
from ..utils.output_parser import get_pydantic_output_parser
from .base_node import BaseNode


class GenerateAnswerNode(BaseNode):
    """
    Initializes the GenerateAnswerNode class.
    ... [Docstring remains same] ...
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "GenerateAnswer",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)
        self.llm_model = node_config["llm_model"]

        if isinstance(node_config["llm_model"], ChatOllama):
            if node_config.get("schema", None) is None:
                self.llm_model.format = "json"
            else:
                self.llm_model.format = self.node_config["schema"].model_json_schema()

        self.verbose = node_config.get("verbose", False)
        self.force = node_config.get("force", False)
        self.script_creator = node_config.get("script_creator", False)
        self.is_md_scraper = node_config.get("is_md_scraper", False)
        self.additional_info = node_config.get("additional_info")
        self.timeout = node_config.get("timeout", 480)

    def invoke_with_timeout(self, chain, inputs, timeout):
        """Helper method to invoke chain with timeout"""
        try:
            start_time = time.time()
            response = chain.invoke(inputs)
            if time.time() - start_time > timeout:
                raise Timeout(f"Response took longer than {timeout} seconds")
            return response
        except Timeout as e:
            self.logger.error(f"Timeout error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during chain execution: {str(e)}")
            raise

    def process(self, state: dict) -> dict:
        """Process the input state and generate an answer."""
        user_prompt = state.get("user_prompt")
        # Check for content in different possible state keys
        content = (
            state.get("relevant_chunks")
            or state.get("parsed_doc")
            or state.get("doc")
            or state.get("content")
        )

        if not content:
            raise ValueError("No content found in state to generate answer from")

        if not user_prompt:
            raise ValueError("No user prompt found in state")

        # Create the chain input with both content and question keys
        chain_input = {"content": content, "question": user_prompt}

        try:
            response = self.invoke_with_timeout(self.chain, chain_input, self.timeout)
            state.update({self.output[0]: response})
            return state
        except Exception as e:
            self.logger.error(f"Error in GenerateAnswerNode: {str(e)}")
            raise

    def execute(self, state: dict) -> dict:
        """
        Executes the GenerateAnswerNode.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]
        user_prompt = input_data[0]
        doc = input_data[1]

        # 1. Setup Output Parser (Used by both standard and agent flow)
        if self.node_config.get("schema", None) is not None:
            # Logic to setup Pydantic output parser based on LLM type
            # ... [Paste your existing output parser setup logic here] ...
            if isinstance(self.llm_model, ChatOpenAI):
                output_parser = get_pydantic_output_parser(self.node_config["schema"])
                format_instructions = output_parser.get_format_instructions()
            else:
                if not isinstance(self.llm_model, ChatBedrock):
                    output_parser = get_pydantic_output_parser(
                        self.node_config["schema"]
                    )
                    format_instructions = output_parser.get_format_instructions()
                else:
                    output_parser = None
                    format_instructions = ""
        else:
            # Logic to setup generic JSON output parser
            # ... [Paste your existing JsonOutputParser setup logic here] ...
            if not isinstance(self.llm_model, ChatBedrock):
                output_parser = JsonOutputParser()
                format_instructions = (
                    "You must respond with a JSON object. Your response should be formatted as a valid JSON "
                    "with a 'content' field containing your analysis. For example:\n"
                    '{{"content": "your analysis here"}}'
                )
            else:
                output_parser = None
                format_instructions = ""

        # 2. Setup Prompts
        # ... [Paste your existing prompt template setup logic here] ...
        if (
            not self.script_creator
            or self.force
            and not self.script_creator
            or self.is_md_scraper
        ):
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS_MD
            template_chunks_prompt = TEMPLATE_CHUNKS_MD
            template_merge_prompt = TEMPLATE_MERGE_MD
        else:
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS
            template_chunks_prompt = TEMPLATE_CHUNKS
            template_merge_prompt = TEMPLATE_MERGE

        if self.additional_info is not None:
            template_no_chunks_prompt = self.additional_info + template_no_chunks_prompt
            template_chunks_prompt = self.additional_info + template_chunks_prompt
            template_merge_prompt = self.additional_info + template_merge_prompt


        # 3. Execution Logic
        tools = self.node_config.get("tools", [])
        
        # We only integrate the AGENTIC FLOW for the primary extraction task (single chunk)
        if len(doc) == 1 and tools:
            # ============================================================
            # 🤖 AGENTIC FLOW (Single Chunk with Tool Calling)
            # ============================================================
            self.logger.info(f"--- [GenerateAnswerNode] Agent Mode Active. Tools: {[t.name for t in tools]} ---")
            
            # 1. Bind Tools and Define Initial Prompt (Goal: Extract/Validate/Generate JSON)
            llm_with_tools = self.llm_model.bind_tools(tools)
            
            # We must instruct the LLM to use the tool and then output the final JSON matching the Pydantic schema
            prompt_template_content = template_no_chunks_prompt.format(
                content=doc, 
                question=user_prompt, 
                format_instructions=format_instructions
            )
            
            sys_msg = SystemMessage(
                content="You are a precise extraction and validation agent. "
                        "Your final output MUST be a JSON object that strictly adheres to the format instructions provided. "
                        "You MUST verify company listings using `check_bursa_listing` before finalizing the extraction."
                        "Do not include the news for tickers/names that failed the validation from `check_bursa_listing` tool."
            )
            human_msg = HumanMessage(content=prompt_template_content)
            messages = [sys_msg, human_msg]
            
            # 2. Execution Loop
            max_turns = 10
            final_response_str = None
            
            for turn in range(max_turns):
                try:
                    # Invoke LLM (expecting a tool call or final text response)
                    ai_msg = self.invoke_with_timeout(llm_with_tools, messages, self.timeout)
                    messages.append(ai_msg)
                    
                    if ai_msg.tool_calls:
                        self.logger.info(f"--- [Agent] Turn {turn+1}: LLM requesting {len(ai_msg.tool_calls)} tool calls ---")
                        
                        tool_outputs = []
                        for tool_call in ai_msg.tool_calls:
                            selected_tool = next((t for t in tools if t.name == tool_call["name"]), None)
                            if selected_tool:
                                try:
                                    tool_output = selected_tool.invoke(tool_call["args"])
                                    self.logger.info(f"    > Tool '{tool_call['name']}' Output: {str(tool_output)[:100]}...")
                                except Exception as e:
                                    tool_output = f"Error executing tool: {e}"
                                
                                tool_outputs.append(ToolMessage(
                                    content=str(tool_output),
                                    tool_call_id=tool_call["id"]
                                ))
                            else:
                                tool_outputs.append(ToolMessage(content="Error: Tool not found.", tool_call_id=tool_call["id"]))
                        messages.extend(tool_outputs)

                    else:
                        # No tool calls -> Final Answer is generated
                        final_response_str = ai_msg.content
                        break
                        
                except Exception as e:
                    self.logger.error(f"--- [Agent] Error in loop: {e}")
                    # If the agent loop fails, we fallback to the error handler outside
                    raise

            # 4. ✅ FINAL STEP: Apply Output Parser to the final string
            if final_response_str:
                try:
                    # Clean up JSON if LLM wrapped it in markdown
                    if isinstance(final_response_str, str):
                        import re
                        json_match = re.search(r"```json\s*(\{.*?\})\s*```", final_response_str, re.DOTALL)
                        if json_match:
                            final_response_str = json_match.group(1)
                    
                    answer = json.loads(final_response_str)

                    # Now that we have the dict, validate structure if output_parser is set
                    if output_parser:
                        answer = output_parser.parse(json.dumps(answer))

                except Exception as e:
                    error_msg = f"Final Answer Parsing Failed: {str(e)}"
                    state.update({self.output[0]: {"error": error_msg, "raw_response": final_response_str}})
                    return state

                state.update({self.output[0]: answer})
                return state

        # ============================================================
        # 📜 STANDARD FLOW (Legacy Chains - Fallback/Multi-Chunk)
        # ============================================================
        
        # [The original code block now runs only if tools is False OR if len(doc) > 1]
        
        if len(doc) == 1:
            prompt = PromptTemplate(
                template=template_no_chunks_prompt,
                input_variables=["content", "question"],
                partial_variables={
                    "format_instructions": format_instructions,
                },
            )
            chain = prompt | self.llm_model
            if output_parser:
                chain = chain | output_parser

            try:
                answer = self.invoke_with_timeout(
                    chain, {"content": doc, "question": user_prompt}, self.timeout
                )
            except (Timeout, json.JSONDecodeError) as e:
                error_msg = (
                    "Response timeout exceeded"
                    if isinstance(e, Timeout)
                    else "Invalid JSON response format"
                )
                state.update(
                    {self.output[0]: {"error": error_msg, "raw_response": str(e)}}
                )
                return state

            state.update({self.output[0]: answer})
            return state

        # [Existing Logic: Multi-Chunk (Map-Reduce) Processing]
        chains_dict = {}
        for i, chunk in enumerate(
            tqdm(doc, desc="Processing chunks", disable=not self.verbose)
        ):
            prompt = PromptTemplate(
                template=template_chunks_prompt,
                input_variables=["question"],
                partial_variables={
                    "content": chunk,
                    "chunk_id": i + 1,
                    "format_instructions": format_instructions,
                },
            )
            chain_name = f"chunk{i + 1}"
            chains_dict[chain_name] = prompt | self.llm_model
            if output_parser:
                chains_dict[chain_name] = chains_dict[chain_name] | output_parser

        async_runner = RunnableParallel(**chains_dict)
        try:
            batch_results = self.invoke_with_timeout(
                async_runner, {"question": user_prompt}, self.timeout
            )
        except (Timeout, json.JSONDecodeError) as e:
            error_msg = (
                "Response timeout exceeded during chunk processing"
                if isinstance(e, Timeout)
                else "Invalid JSON response format in chunk processing"
            )
            state.update({self.output[0]: {"error": error_msg, "raw_response": str(e)}})
            return state

        merge_prompt = PromptTemplate(
            template=template_merge_prompt,
            input_variables=["content", "question"],
            partial_variables={"format_instructions": format_instructions},
        )

        merge_chain = merge_prompt | self.llm_model
        if output_parser:
            merge_chain = merge_chain | output_parser
        try:
            answer = self.invoke_with_timeout(
                merge_chain,
                {"content": batch_results, "question": user_prompt},
                self.timeout,
            )
        except (Timeout, json.JSONDecodeError) as e:
            error_msg = (
                "Response timeout exceeded during merge"
                if isinstance(e, Timeout)
                else "Invalid JSON response format during merge"
            )
            state.update({self.output[0]: {"error": error_msg, "raw_response": str(e)}})
            return state

        state.update({self.output[0]: answer})
        return state