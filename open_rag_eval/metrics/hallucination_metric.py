from typing import Dict, Optional
import torch
import warnings

from transformers import AutoModelForSequenceClassification
from open_rag_eval.metrics.base_metrics import AugmentedGenerationMetric
from open_rag_eval.data_classes.rag_results import RAGResult
from open_rag_eval.models.llm_judges import OpenAIModel

# Set number of cores to 2 to avoid heavy CPU usage
torch.set_num_threads(2)

class HallucinationMetric(AugmentedGenerationMetric):
    """ This metric uses the Vectara Hallucination Evaluation Model to detect hallucinations in RAG output. """

    def __init__(self, model_name: str = 'vectara/hallucination_evaluation_model', detection_threshold: float = 0.5, max_chars: int = 8192):
        """Initialize the Hallucination metric.

        Args:
            model_name (str): The name of the model to use for hallucination detection.
            detection_threshold (float): The threshold fordetecting hallucinations.
            max_chars (int): The maximum number of characters to process. Inputs longer than this will be truncated.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                revision="main"
            )
        self.detection_threshold = detection_threshold

    def compute(self, rag_result: RAGResult) -> Dict[str, int]:
        # Create source and summary pair.
        passage_text_collection = []
        retrieval_results = rag_result.retrieval_result
        for _, passage in retrieval_results.retrieved_passages.items():
            passage_text_collection.append(passage)

        summary_text_collection = [generated_answer_part.text for generated_answer_part in rag_result.generation_result.generated_answer]

        sources = " ".join(passage_text_collection)
        summary = " ".join(summary_text_collection)

        # HHEM-2.1-Open supports unlimited context length, no truncation needed
        # Suppress token length warnings during inference as HHEM-2.1-Open handles longer sequences correctly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
            # Call the hallucination detection model.
            score = self.model.predict([(sources, summary)]).item()

        return {"hhem_score": score}


class GPTHallucinationMetric(AugmentedGenerationMetric):
    """ This metric uses GPT-4o through OpenRouter API to detect hallucinations in RAG output. """

    def __init__(self, model_name: str = "qwen-plus", detection_threshold: float = 0.5):
        """Initialize the GPT Hallucination metric.

        Args:
            model_name (str): The model name to use. Defaults to 'qwen-plus'.
            detection_threshold (float): The threshold for detecting hallucinations.
        """
        self.model = OpenAIModel({
            "name": model_name,
            "api_key": "sk-6a44d15e56dd4007945ccc41b97b499c",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        })
        self.detection_threshold = detection_threshold

    def compute(self, rag_result: RAGResult) -> Dict[str, float]:
        # Create source and summary pair
        passage_text_collection = []
        retrieval_results = rag_result.retrieval_result
        for _, passage in retrieval_results.retrieved_passages.items():
            passage_text_collection.append(passage)

        summary_text_collection = [generated_answer_part.text for generated_answer_part in rag_result.generation_result.generated_answer]

        sources = " ".join(passage_text_collection)
        summary = " ".join(summary_text_collection)

        # Create hallucination detection prompt
        prompt = f"""You are an expert at detecting hallucinations in AI-generated responses. Your task is to determine if a generated answer is factually consistent with the provided source material.

Please analyze the following:

**Source Material:**
{sources}

**Generated Answer:**
{summary}

**Instructions:**
1. Carefully compare the generated answer with the source material
2. Evaluate how well the generated answer is supported by the source material
3. Pay special attention to:
   - Factual claims and whether they are supported by sources
   - Numbers, dates, or statistics and their accuracy
   - Information that goes beyond what the sources state
   - Claims that seem plausible but aren't actually in the sources

**Response Format:**
Provide a score between 0 and 1, where:
- 1 = Fully supported (answer is completely consistent with and supported by sources)
- 0 = Not supported (answer contains significant information not found in sources)

This scoring system matches the Vectara HHEM model where 1 indicates full factual consistency and 0 indicates lack of support.

Respond with only the numerical score (e.g., 0.8)."""

        try:
            response = self.model.call(prompt)
            score_str = response.strip()
            score = float(score_str)
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))

            return {"hhem_score": score}
        except (ValueError, Exception):
            # If parsing fails, return a neutral score
            return {"hhem_score": 0.5}
