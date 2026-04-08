from python:3.9-slim

WORKDIR /

COPY . .

RUN pip install -r reqirements.txt

CMD [ "python" , "app.py"]