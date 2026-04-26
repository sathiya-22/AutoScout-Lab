FROM python:3.10-slim-buster

WORKDIR /app

# Install system dependencies required for audio/video processing and building Python packages
# ffmpeg: Essential for handling media files (e.g., extracting audio, processing video).
# libsndfile1: Required by the 'soundfile' library for reading/writing audio.
# build-essential: Necessary for compiling certain Python packages that have C/C++ extensions (e.g., pyworld).
# git: Useful for installing packages directly from source repositories or cloning models.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache for dependencies
# Assuming requirements.txt exists at the project root with all necessary Python packages.
# Example content for requirements.txt (not part of Dockerfile output, but for context):
# numpy
# scipy
# librosa
# pydub
# ffmpeg-python
# demucs # This pulls torch and its dependencies for source separation
# crepe
# pyworld
# opencv-python
# fastapi
# uvicorn[standard]
# python-multipart
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to prevent pip from storing cache, reducing image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Ensure the 'models' directory exists. Pre-trained models might be downloaded
# by the application at runtime, or mounted as a volume.
# This command ensures the directory structure is ready.
RUN mkdir -p /app/models

# Expose the port on which the FastAPI/Flask service will run.
# Default for FastAPI is 8000.
EXPOSE 8000

# Define the default command to run when the container starts.
# This command starts the FastAPI application using Uvicorn.
# It assumes 'src/api/web_service.py' contains a FastAPI instance named 'app'.
CMD ["uvicorn", "src.api.web_service:app", "--host", "0.0.0.0", "--port", "8000"]