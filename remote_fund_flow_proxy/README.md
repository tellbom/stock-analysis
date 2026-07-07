# remote_fund_flow_proxy

Lightweight FastAPI proxy for Eastmoney historical fund flow data.

## Deploy

```bash
# On remote server
cd remote_fund_flow_proxy
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 18088
```

## API

### Health check

```bash
curl http://<host>:18088/health
```

### Single stock

```bash
curl http://<host>:18088/fund-flow/000001
curl http://<host>:18088/fund-flow/600000
```

### Batch

```bash
curl -X POST http://<host>:18088/fund-flow/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols":["600000","000001","300750","688981"],"sleep_seconds":1.0}'
```

## Local client usage

```python
import requests

# Single stock
r = requests.get("http://192.168.124.2:18088/fund-flow/000001")
data = r.json()

# Batch
r = requests.post("http://192.168.124.2:18088/fund-flow/batch",
    json={"symbols": ["600000", "000001", "300750", "688981"]})
data = r.json()
```
