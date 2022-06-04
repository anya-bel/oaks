FROM python:3.9.1
ADD . /python-flask
WORKDIR /python-flask/app/frontend/website
RUN pip3 install torch==1.10.1 
RUN pip3 install -r requirements.txt
