
FROM python:3.13.15-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Embeddings are served by the Great Learning OpenAI-compatible API at runtime
# (AGENTS.md section 6.16). EMBEDDING_API_KEY is passed with `docker run -e`;
# the base URL and model have code defaults in app.py. No local embedding
# weights are downloaded, so no torch / HF offline env is needed.

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
