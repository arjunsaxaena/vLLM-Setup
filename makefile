run:
	uv run python src/test.py

serve:
	uv run uvicorn src.server:app --host 0.0.0.0 --port 8000