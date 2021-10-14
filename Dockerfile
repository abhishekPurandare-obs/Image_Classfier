FROM mlflow-base:latest
WORKDIR /opt/program
RUN python -c 'import mlflow; print(mlflow.__version__)'
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ENV MLFLOW_TRACKING_URI=http://ec2-13-232-139-2.ap-south-1.compute.amazonaws.com:5000/
ENV MLFLOW_EXP_NAME=Experiment-abhishek

COPY . /opt/program/
RUN chmod +x project.env
RUN chmod +x exp_env.sh
RUN chmod +x train
# RUN chmod +x serve
EXPOSE 5000
