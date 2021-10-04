FROM python:3.8
RUN apt-get update
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c 'import mlflow; print(mlflow.__version__)'
EXPOSE 5000
