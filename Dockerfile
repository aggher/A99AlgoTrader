# Institutional Trading System v2: Backend Dockerfile
FROM python:3.11-slim

# 1. Install System Dependencies (for TA-Lib and technical analysis)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Install TA-Lib C-Library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# 3. Set Working Directory
WORKDIR /app

# 4. Copy Requirements and Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Create Data Directory (for persistent SQLite volume)
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:////data/trading.db

# 7. Expose Port
EXPOSE 8000

# 8. Entrypoint: Run both API and Scheduler
CMD ["python", "launch.py"]
