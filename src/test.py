try:
    from src.inference import VLLMInference
except ImportError:
    from inference import VLLMInference

if __name__ == "__main__":
    model = VLLMInference()

    result = model.generate("Explain machine learning in simple words")
    print(result["response"])
    print(f"Generated in {result['elapsed_ms']} ms")