# Base image
FROM python:3.12.0b1-slim-buster

# Set the working directory
WORKDIR /app

# Copy requirements.txt before other files to leverage Docker caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . /app

# Expose the default Streamlit port
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "sample_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
