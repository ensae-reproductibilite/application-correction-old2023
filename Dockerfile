FROM ubuntu:22.04

WORKDIR ${HOME}/titanic

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py .
COPY src ./src
CMD ["python3", "train.py"]