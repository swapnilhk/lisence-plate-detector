FROM ubuntu:22.04
RUN apt-get update && apt-get install python3 python3-pip ffmpeg libsm6 libxext6 -y
WORKDIR /usr/src/app
COPY src .
RUN pip3 install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --default-timeout=1000 -r requirements.txt --verbose
RUN pip3 install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --default-timeout=1000 -r yolov6/requirements.txt --verbose
RUN pip3 install requests
RUN pip3 install psutil
CMD python3 main.py
