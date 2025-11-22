FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Expose port for Prometheus metrics and API
EXPOSE 9090
EXPOSE 5000

# Default command to start the API; override with docker-compose if training
CMD ["python3", "-m", "capsule_brain.api.app"]