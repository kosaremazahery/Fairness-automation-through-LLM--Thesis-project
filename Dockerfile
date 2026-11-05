FROM python:3.11-slim

ENV OLLAMA_HOST=http://host.docker.internal:11434

# set a working directory
WORKDIR /app

# copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your code
COPY . .

# default command: run main.py
CMD ["python", "main.py"]