FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.pkl .

# Tell Docker this container will be reachable on port 8000
EXPOSE 8000

# Unlike train.py (which ran once and exited), uvicorn STAYS RUNNING,
# continuously listening for requests
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
