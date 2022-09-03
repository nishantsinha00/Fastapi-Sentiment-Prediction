FROM python:3.7
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./tokenizer.pickle /app
COPY ./config.json /app
COPY ./model /app/model
COPY ./app /app/app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]