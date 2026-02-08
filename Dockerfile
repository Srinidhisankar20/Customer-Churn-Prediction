FROM python:3.10-slim-bookworm


# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Fix apt issue and clean up to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python3", "app.py"]
