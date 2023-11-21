FROM ubuntu:22.04

WORKDIR /app

COPY resources/ ./resources/
ADD app.py .
ADD requirements.txt .

RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN apt-get install libdmtx0b
RUN pip install -r ./requirements.txt

CMD ["python", "./app.py"]
