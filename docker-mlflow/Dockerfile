# FROM: Start with a kitchen that already has Python 3.11 pre-installed
FROM python:3.11-slim

# WORKDIR: Set up our workspace folder inside the container
WORKDIR /app

# COPY: Bring requirements.txt in FIRST (explained below why order matters)
COPY requirements.txt .

# RUN: Install our packages ONCE, while building the image
RUN pip install --no-cache-dir -r requirements.txt

# COPY: Now bring in our actual training script
COPY train.py .

# CMD: When the container starts, run this command
CMD ["python", "train.py"]
