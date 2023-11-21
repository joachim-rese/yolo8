FROM ubuntu:22.04

WORKDIR /app

COPY resources/ ./resources/
ADD app.py .
ADD requirements.txt .

#RUN apt-get update -y
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN apt-get install libdmtx0b
RUN pip install -r ./requirements.txt
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install python3-opencv
RUN pip install opencv-python

CMD ["python3", "./app.py"]
