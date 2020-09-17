ARG CPU_OR_GPU
ARG AWS_REGION
ARG FRAMEWORK
FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-${FRAMEWORK}-${CPU_OR_GPU}-py36

WORKDIR /opt/ml
RUN apt-get update
RUN apt-get install -y openssh-server
RUN pip install --upgrade \
    pip \
    setuptools \
    setproctitle \
    lz4 \
    psutil
    
RUN pip install procgen==0.10.1
RUN pip install mlflow==1.8.0

# Log in to dockerized SSH daemon service
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/^#?PubkeyAuthentication\s+.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train; import ray; from sagemaker_containers.cli.train import main"
# Make things a bit easier to debug
WORKDIR /opt/ml/code
# Expose port 22 for SSH login
EXPOSE 22
CMD    ["/usr/sbin/sshd", "-D"]
