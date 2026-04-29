FROM python:3.11-slim

RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN g++ -O3 -std=c++17 -o engines/engine engine_src/engine.cpp -lpthread
RUN chmod +x engines/engine

EXPOSE 8080
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]