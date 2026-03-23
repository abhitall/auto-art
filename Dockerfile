FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev --no-install-project

COPY auto_art ./auto_art

RUN uv sync --frozen --no-dev

EXPOSE 5000

ENV FLASK_APP=auto_art.api.app:app

CMD ["uv", "run", "flask", "run", "--host=0.0.0.0", "--port=5000"]
