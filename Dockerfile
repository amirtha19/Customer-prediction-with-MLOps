# Use an official Linux distribution as a base image
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /app

# Install required packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y libopenblas-dev liblapack-dev && \
    apt-get install -y python3-dev

# Install Python packages
RUN pip3 install zenml["server"] pandas pyxlsb openpyxl scikit-learn mlflow

# Copy your files into the container
COPY . /app
COPY data/data.xlsx /app/data/

# Register ZenML components
RUN zenml experiment-tracker register mlflow_tracker --flavor=mlflow
RUN zenml model-deployer register mlflow --flavor=mlflow
RUN zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set


CMD ["python3", "run_deployment.py", "--config", "deploy_and_predict"]
