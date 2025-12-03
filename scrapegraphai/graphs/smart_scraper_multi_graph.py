"""
SmartScraperMultiGraph Module
"""

from copy import deepcopy
from typing import List, Optional, Type

from pydantic import BaseModel

from ..nodes import GraphIteratorNode, MergeAnswersNode
from ..utils.copy import safe_deepcopy
from .abstract_graph import AbstractGraph
from .base_graph import BaseGraph
from .smart_scraper_graph import SmartScraperGraph


class SmartScraperMultiGraph(AbstractGraph):
    """
    SmartScraperMultiGraph is a scraping pipeline that scrapes a
    list of URLs and generates answers to a given prompt.
    It only requires a user prompt and a list of URLs.
    The difference with the SmartScraperMultiLiteGraph is that in this case the content will be abstracted
    by llm and then merged finally passed to the llm.

    Attributes:
        prompt (str): The user prompt to search the internet.
        llm_model (dict): The configuration for the language model.
        embedder_model (dict): The configuration for the embedder model.
        headless (bool): A flag to run the browser in headless mode.
        verbose (bool): A flag to display the execution information.
        model_token (int): The token limit for the language model.

    Args:
        prompt (str): The user prompt to search the internet.
        source (List[str]): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (Optional[BaseModel]): The schema for the graph output.
        merge (str): "yes" to enable merging answers node, "no" to skip it. Default "no".

    Example:
        >>> smart_scraper_multi_graph = SmartScraperMultiGraph(
        ...     prompt="Who is ?",
        ...     source= [
        ...         "https://perinim.github.io/",
        ...         "https://perinim.github.io/cv/"
        ...     ],
        ...     config={"llm": {"model": "openai/gpt-3.5-turbo"}}
        ... )
        >>> result = smart_scraper_multi_graph.run()
    """

    def __init__(
            self,
            prompt: str,
            source: List[str],
            config: dict,
            schema: Optional[Type[BaseModel]] = None,
            merge: str = "no",
        ):
        # normalize and store merge flag
        self.merge = (merge or "no").strip().lower()
        self.max_results = config.get("max_results", 3)
        
        # --- FIX START ---
        # 1. Back up the tools because safe_deepcopy might strip them
        original_tools = config.get("tools", [])
        
        # 2. Perform deepcopy
        self.copy_config = safe_deepcopy(config)
        
        # 3. Restore tools explicitly if they were lost/corrupted
        if original_tools:
            self.copy_config["tools"] = original_tools
        # --- FIX END ---

        self.copy_schema = deepcopy(schema)

        super().__init__(prompt, config, source, schema)

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for web scraping and searching.

        Returns:
            BaseGraph: A graph instance representing the web scraping and searching workflow.
        """

        # always create iterator node
        graph_iterator_node = GraphIteratorNode(
            input="user_prompt & urls",
            output=["results"],
            node_config={
                "graph_instance": SmartScraperGraph,
                "scraper_config": self.copy_config,
            },
            schema=self.copy_schema,
        )

        # prepare containers and always add iterator node
        nodes = [graph_iterator_node]
        edges = []

        # conditionally add merge node only when merge == "yes"
        if self.merge == "yes":
            merge_answers_node = MergeAnswersNode(
                input="user_prompt & results",
                output=["answer"],
                node_config={"llm_model": self.llm_model, "schema": self.copy_schema},
            )
            nodes.append(merge_answers_node)
            edges.append((graph_iterator_node, merge_answers_node))
        # If merge is not enabled, no merge node or edge is added.

        return BaseGraph(
            nodes=nodes,
            edges=edges,
            entry_point=graph_iterator_node,
            graph_name=self.__class__.__name__,
        )
    
    def run(self) -> str:
        """
        Executes the web scraping and searching process.

        Returns:
            str: The answer to the prompt.
        """

        inputs = {"user_prompt": self.prompt, "urls": self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)
        
        # Debugging print (Optional)
        # print("Printing self.final_state:", self.final_state)

        # --- FIX STARTS HERE ---
        # Priority 1: Check for 'answer' (standard single graph)
        if "answer" in self.final_state:
            return self.final_state["answer"]
            
        # Priority 2: Check for 'results' (multi-graph / batch output)
        if "results" in self.final_state:
            # 'results' is usually a list of dicts, e.g. [{'company': [...]}]
            # We return it directly so the scraper wrapper handles the parsing
            return self.final_state["results"][0]
            
        return "No answer found."
        # --- FIX ENDS HERE ---
