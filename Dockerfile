FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install AWS CLI via pip
RUN pip install --no-cache-dir awscli

CMD ["python3", "app.py"]


