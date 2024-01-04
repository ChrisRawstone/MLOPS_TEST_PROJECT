# Base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir


COPY CNN_Project/ CNN_Project/
COPY data/ data/
COPY reports/ reports/
COPY models/ models/

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "CNN_Project/train_model.py"]

RUN docker cp 5715fbb43247:CNN_Project/models/model.pt models/docker_model.pt