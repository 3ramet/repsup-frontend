FROM python:3.11-slim

# environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# reduce TF logging and prevent TF attempting to use GPUs inside container
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# system deps for some packages (e.g., xgboost, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy application
COPY . /app

# create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Run with Gunicorn + eventlet worker for proper Socket.IO support
# This avoids using Flask's development server (which prints the dev warning)
CMD ["gunicorn", "server:app", "-b", "0.0.0.0:8080", "-k", "eventlet", "-w", "1"]
