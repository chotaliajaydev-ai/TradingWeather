FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["bash", "-lc", "streamlit run tradingweather.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
