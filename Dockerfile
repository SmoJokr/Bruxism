FROM python:3.10-slim

WORKDIR /app

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1

RUN pip install --no-cache-dir numpy==1.23.5 scikit-learn==1.2.1

RUN pip install --no-cache-dir timm

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "CNN_LSTM_ATT_5CROSS.py"]
