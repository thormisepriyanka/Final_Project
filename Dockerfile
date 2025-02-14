FROM python:3.8-slim-buster

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /

RUN pip install -r ./requirements.txt

ENTRYPOINT ["streamlit", "run", "sample_app.py", "--server.port-8501", "--server.address=0.0.0.0"]
