import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
from evaluation_metrics import build_metrics

class EvaluationAgent:
    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.0,
        threshold: float = 0.8,
        task_type: str = "general", 
        enable_logging: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.threshold = threshold
        self.task_type = task_type.lower()
        self.logger = self._init_logger(enable_logging)

        # Initialize custom LLM wrapper
        self.llm = self.CustomOpenAILLM(model_name, temperature)

        # Load metrics
        self.metrics = self._load_metrics()
        self.logger.info(f"Initialized EvaluationAgent for task: {self.task_type}")

    class CustomOpenAILLM(DeepEvalBaseLLM):
        def __init__(self, model: str, temperature: float):
            self.model_name = model
            self.model = ChatOpenAI(model=model, temperature=temperature)

        def generate(self, prompt: str) -> str:
            response = self.model.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        async def a_generate(self, prompt: str) -> str:
            response = await self.model.ainvoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        def load_model(self):
            return self.model

        def get_model_name(self) -> str:
            return self.model_name

    def _load_metrics(self):
        # Simply call the unified build_metrics from evaluation_metrics.py
        return build_metrics(self.llm, self.threshold, self.task_type)

    def _init_logger(self, enable_logging: bool):
        logger = logging.getLogger(f"EvaluationAgent-{id(self)}")
        if enable_logging:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            logger.addHandler(logging.NullHandler())
        return logger

    # ------------------------------------------------------------
    # ✅ FIXED: Evaluation Runner using 'a_measure()'
    # ------------------------------------------------------------
    async def a_evaluate(
        self,
        generated_text: str,
        source_context: str,
        retrieved_chunks: list[str] = None,
        section_topic: str = "General", # We use this to pass the map if needed
    ) -> Dict[str, Any]:
        
        # 1. Prepare Context
        # If section_topic contains "Localization Map", we treat it as context for GEval
        # For Faithfulness, we typically use 'retrieved_chunks'
        
        # If no explicit chunks provided, assume Source Text is the chunk (for summarization)
        final_retrieval = retrieved_chunks if retrieved_chunks else [source_context]
        
        # 2. Build Test Case
        test_case = LLMTestCase(
            input=source_context,
            actual_output=generated_text,
            retrieval_context=final_retrieval,
            context=[section_topic], # Inject map here for GEval to see
            metadata={"section_topic": section_topic},
        )

        # 3. Run Metrics Correctly (Parallel)
        # We define a helper to run a_measure and return the metric itself
        async def run_metric(name, metric_obj):
            try:
                await metric_obj.a_measure(test_case)
                return name, metric_obj
            except Exception as e:
                self.logger.error(f"Error running metric {name}: {e}")
                # return dummy if failed
                metric_obj.score = 0.0
                metric_obj.reason = f"Execution Failed: {str(e)}"
                return name, metric_obj

        # Execute all
        results = await asyncio.gather(
            *[run_metric(name, m) for name, m in self.metrics.items()]
        )

        # 4. Compile Results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "section_topic": section_topic,
            "metrics": {},
        }

        for name, metric_obj in results:
            passed = metric_obj.is_successful() # DeepEval standard method
            score = metric_obj.score
            reason = metric_obj.reason

            summary["metrics"][name] = {
                "score": score,
                "threshold": getattr(metric_obj, "threshold", 0.5),
                "passed": passed,
                "feedback": reason,
            }

            self.logger.info(
                f"[{self.task_type.upper()}] {name}: {score:.3f} | {'PASS' if passed else 'FAIL'}"
            )

        summary["overall_pass"] = all(m["passed"] for m in summary["metrics"].values())
        if summary["metrics"]:
            summary["average_score"] = round(
                sum(m["score"] for m in summary["metrics"].values()) / len(summary["metrics"]),
                3,
            )
        else:
            summary["average_score"] = 0.0

        return summary

    # Sync wrapper just in case
    def evaluate(self, *args, **kwargs):
        return asyncio.run(self.a_evaluate(*args, **kwargs))