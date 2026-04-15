from time import perf_counter

from vllm import LLM, SamplingParams
from config import get_model_name


class VLLMInference:
    def __init__(self, model_name=None):
        model_name = model_name or get_model_name()
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 200,
    ):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        start_time = perf_counter()
        outputs = self.llm.generate(prompt, sampling_params)
        elapsed_ms = (perf_counter() - start_time) * 1000

        return {
            "response": outputs[0].outputs[0].text,
            "elapsed_ms": round(elapsed_ms, 2),
        }