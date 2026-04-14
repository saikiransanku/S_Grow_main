FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=ai_backend.settings
ENV DJANGO_ALLOWED_HOSTS=*
ENV CORS_ALLOW_ORIGIN=*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "ai_backend.wsgi:application", "--bind", "0.0.0.0:7860", "--timeout", "120"]
