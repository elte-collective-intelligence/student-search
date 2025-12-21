FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml /app

RUN pip install --no-cache-dir --break-system-packages .

COPY . /app

# TensorBoard default port
EXPOSE 6006

ENTRYPOINT ["python", "-m", "src.main"]

CMD ["train.active=true"]
