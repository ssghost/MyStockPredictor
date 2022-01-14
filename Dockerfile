FROM rocm/tensorflow:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# If you have an IEX key, run this code:
# RUN /bin/sh -c 'python train.py && python test.py'