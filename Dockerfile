# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or install directly
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pydantic \
    openai \
    gradio

# Copy the environment code
COPY . .

# Set environment variables for OpenEnv
ENV PYTHONUNBUFFERED=1

# The entrypoint can be a simple script to run inference
# or a command to start a web server if the Spaces requires one.
# For OpenEnv, we primarily need the environment to be accessible.
EXPOSE 7860

CMD ["python", "app.py"]
