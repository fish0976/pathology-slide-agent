FROM node:22-bookworm-slim AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY pathology/ ./pathology/
RUN pip install --no-cache-dir '.[wsi]'
COPY --from=frontend /ui/dist ./frontend/dist
RUN useradd --create-home appuser && mkdir /app/data && chown appuser:appuser /app/data
USER appuser
ENV DATA_DIR=/app/data
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "pathology.main:app", "--host", "0.0.0.0", "--port", "8000"]
