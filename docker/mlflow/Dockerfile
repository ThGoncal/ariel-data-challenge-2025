FROM python:3.10-slim

RUN pip install mlflow boto3 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]
