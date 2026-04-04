# Використовуємо легкий офіційний образ Python 3.11
FROM python:3.11-slim

# Створюємо користувача "user" з ID 1000 (обов'язкова вимога Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Встановлюємо робочу директорію
WORKDIR /app

# Встановлюємо uv
RUN pip install --no-cache-dir uv

# Копіюємо конфігурацію залежностей та встановлюємо їх
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen

# Копіюємо всі модулі додатку та папку зі стилями
COPY --chown=user:user app.py ui_layout.py server.py plots.py utils.py ./
COPY --chown=user:user pvalue_ui.py pvalue_server.py pvalue_plots.py ./
COPY --chown=user:user css/ ./css/

# Відкриваємо порт
EXPOSE 7860

# Запускаємо Shiny додаток
CMD ["uv", "run", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]