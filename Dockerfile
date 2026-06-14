# Minimal image for the Phase 1 vertical slice. Phase 5 hardens this:
# multi-stage/distroless, non-root, read-only rootfs, dropped capabilities.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "python", "-m", "agent"]
