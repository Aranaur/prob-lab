# Використовуємо легкий офіційний образ Python 3.11
FROM python:3.11-slim

# Встановлюємо uv через копіювання бінарника (надійніше)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Створюємо користувача "user" з ID 1000 (обов'язкова вимога Hugging Face)
RUN useradd -m -u 1000 user

# Встановлюємо робочу директорію та даємо права користувачу
WORKDIR /app
RUN chown user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Налаштування uv для роботи в Docker (запобігає зависанням через hardlinks)
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Копіюємо конфігурацію залежностей та встановлюємо їх
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen

# Копіюємо всі модулі додатку та папку зі стилями
COPY --chown=user:user *.py ./

COPY --chown=user:user css/ ./css/

# Відкриваємо порт
EXPOSE 7860

# Запускаємо Shiny додаток
CMD ["uv", "run", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]