FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (swap for GPU variant if needed)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install ReOldify
COPY . .
RUN pip install --no-cache-dir -e .

# Create models directory
RUN mkdir -p /app/models

EXPOSE 8000

# Default: run the API server
CMD ["python", "-m", "deoldify.api"]
