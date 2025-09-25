FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update

WORKDIR /app

COPY  . /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# COPY . /app
# RUN python main.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
