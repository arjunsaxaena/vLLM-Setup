from time import perf_counter

from vllm import LLM, SamplingParams

class VLLMInference:
    def __init__(self, model_name="Qwen/Qwen2-0.5B-Instruct"):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.8,
            max_model_len=2048
        )

    def generate(self, prompt):
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=200
        )

        start_time = perf_counter()
        outputs = self.llm.generate(
            prompt, sampling_params
        )
        elapsed_ms = (perf_counter() - start_time) * 1000

        return {
            "response": outputs[0].outputs[0].text,
            "elapsed_ms": round(elapsed_ms, 2),
        }