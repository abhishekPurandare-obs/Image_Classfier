FROM python:3.8
RUN apt-get update
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c 'import mlflow; print(mlflow.__version__)'
EXPOSE 5000
