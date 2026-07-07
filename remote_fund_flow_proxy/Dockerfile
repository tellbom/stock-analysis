FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install playwright urllib3

# Install chromium + system deps via playwright
RUN python -m playwright install chromium
RUN python -m playwright install-deps chromium

# Install cloakbrowser from local source
COPY cloakbrowser_src /tmp/cloakbrowser_src
RUN pip install -e /tmp/cloakbrowser_src

WORKDIR /app
COPY app_cloak.py /app/app.py

EXPOSE 18088
CMD ["python", "app.py"]
