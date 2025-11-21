FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r cyborg_mind_v2/deployment/requirements.txt

# Expose port for monitoring (Prometheus default)
EXPOSE 9090

# Default command (placeholder)
CMD ["python3", "-m", "cyborg_mind_v2.training.teacher_student_trainer_real"]