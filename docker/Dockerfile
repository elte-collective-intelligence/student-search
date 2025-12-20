FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY . /app

ENTRYPOINT ["python", "-m", "src.main"]

CMD ["train.active=true"]
