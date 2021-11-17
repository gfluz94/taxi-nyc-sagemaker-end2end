FROM python:3.7-slim-buster

RUN apt-get -y update && apt-get install -y --no-install-recommends \
        wget \
        nginx \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

COPY src/* /opt/ml/

ENV PATH="/opt/ml:${PATH}"
WORKDIR /opt/ml/

RUN chmod u+x /opt/ml/train
RUN chmod u+x /opt/ml/serve
EXPOSE 80
EXPOSE 5000

RUN export FLASK_RUN_PORT=80