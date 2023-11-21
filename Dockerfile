FROM ubuntu:22.04

COPY resources/ resources/
ADD app.py .
ADD requirements.txt .

RUN apt-get install libdmtx0b
RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
