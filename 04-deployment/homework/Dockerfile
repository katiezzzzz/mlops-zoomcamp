FROM python:3.13.1-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy

RUN mkdir models
COPY ./models ./models
COPY [ "starter.py", "./" ]
RUN mkdir output

ENTRYPOINT [ "python", "starter.py", "yellow", "2023", "5" ]
