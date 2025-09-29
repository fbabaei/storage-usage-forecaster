# Base image
FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 7860

# Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start Streamlit
CMD ["streamlit", "run", "app.py"]
