FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cryptography "paho-mqtt>=2.1,<3"
WORKDIR /app
COPY protocol.py policy.py ./
COPY scenarios/*.py ./
