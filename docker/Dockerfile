FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY . /app

ENTRYPOINT ["python", "src/main.py"]

CMD ["train.active=true"]
