FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENTRYPOINT ["python", "main_updated.py"]

CMD ["train.active=true"]
