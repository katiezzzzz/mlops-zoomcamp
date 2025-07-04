FROM python:3.13.1-slim

RUN pip install -U pip

COPY [ "requirements.txt", "./" ]
RUN pip install -r requirements.txt

COPY [ "predict.py", "lin_reg.bin", "./" ]

# open port
EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
