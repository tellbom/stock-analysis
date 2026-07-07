"""
Lightweight Eastmoney historical fund flow HTTP proxy.

Deploy on a machine that can reach push2his.eastmoney.com reliably.
Local clients call this proxy instead of hitting Eastmoney directly.

Start:
    python3 app.py
"""

import json
import logging
import time
import sys
from wsgiref.simple_server import make_server

import urllib3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EM_BASE_URL = "http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
EM_FIELDS1 = "f1,f2,f3,f7"
EM_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
EM_LMT = 100000
EM_KLT = 101

FIELD_NAMES = [
    "trade_date",
    "main_net",
    "small_net",
    "medium_net",
    "large_net",
    "super_net",
    "main_net_rate",
    "small_net_rate",
    "medium_net_rate",
    "large_net_rate",
    "super_net_rate",
    "close",
    "pct_change",
]

MARKET_PREFIXES = [
    (("600", "601", "603", "605", "688"), 1),
    (("000", "001", "002", "003", "300", "301"), 0),
    (("430", "830", "831", "832", "833", "834", "835", "836", "837",
      "838", "839", "870", "871", "872", "873", "874", "875", "876",
      "877", "878", "879", "880", "881", "882", "883", "884", "885",
      "886", "887", "888", "889", "920", "921", "922", "923", "924",
      "925", "926", "927", "928", "929"), 2),
]

MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fund_flow_proxy")

http = urllib3.PoolManager(num_pools=1, maxsize=1, retries=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_market(symbol):
    symbol = symbol.strip().upper()
    if "." in symbol:
        code, suffix = symbol.split(".", 1)
        if suffix in ("SH", "1"):
            return code, 1
        elif suffix in ("SZ", "0"):
            return code, 0
        elif suffix in ("BJ", "2"):
            return code, 2
        else:
            raise ValueError("Unknown suffix: %s" % suffix)
    code = symbol
    for prefixes, mkt in MARKET_PREFIXES:
        for pfx in prefixes:
            if code.startswith(pfx):
                return code, mkt
    raise ValueError("Unknown symbol prefix: %s" % symbol)


def parse_kline(kline_str, symbol):
    parts = kline_str.split(",")
    record = {"symbol": symbol}
    for i, name in enumerate(FIELD_NAMES):
        if i >= len(parts):
            break
        val = parts[i]
        if name == "trade_date":
            record[name] = val
        else:
            record[name] = float(val) if val else None
    return record


def fetch_one(symbol):
    code, mkt = resolve_market(symbol)
    secid = "%s.%s" % (mkt, code)
    url = "%s?lmt=%s&klt=%s&secid=%s&fields1=%s&fields2=%s" % (
        EM_BASE_URL, EM_LMT, EM_KLT, secid, EM_FIELDS1, EM_FIELDS2
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            r = http.request("GET", url, timeout=30.0, retries=0)
            if r.status != 200:
                last_error = "HTTP %d" % r.status
                logger.warning("symbol=%s secid=%s attempt=%d status=%d", symbol, secid, attempt + 1, r.status)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                continue

            data = json.loads(r.data.decode("utf-8"))
            if data.get("rc") != 0:
                last_error = "API rc=%s" % data.get("rc")
                logger.warning("symbol=%s secid=%s attempt=%d rc=%s", symbol, secid, attempt + 1, data.get("rc"))
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                continue

            klines = data.get("data", {}).get("klines")
            if not klines:
                logger.info("symbol=%s secid=%s success=True rows=0", symbol, secid)
                return {
                    "symbol": code, "success": True, "rows": 0,
                    "min_date": None, "max_date": None,
                    "source": "eastmoney_push2his_http_remote",
                    "data": [], "error": None,
                }

            records = [parse_kline(k, code) for k in klines]
            min_date = records[0]["trade_date"]
            max_date = records[-1]["trade_date"]
            n = len(records)

            logger.info("symbol=%s secid=%s success=True rows=%d min_date=%s max_date=%s",
                        code, secid, n, min_date, max_date)

            return {
                "symbol": code, "success": True, "rows": n,
                "min_date": min_date, "max_date": max_date,
                "source": "eastmoney_push2his_http_remote",
                "data": records, "error": None,
            }

        except Exception as e:
            last_error = "%s: %s" % (type(e).__name__, e)
            logger.warning("symbol=%s secid=%s attempt=%d/%d error=%s",
                           symbol, secid, attempt + 1, MAX_RETRIES, last_error)

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])

    logger.error("symbol=%s secid=%s success=False error=%s", symbol, secid, last_error)
    return {
        "symbol": code, "success": False, "rows": 0,
        "min_date": None, "max_date": None,
        "source": "eastmoney_push2his_http_remote",
        "data": [], "error": last_error,
    }


# ---------------------------------------------------------------------------
# WSGI Application
# ---------------------------------------------------------------------------

def json_response(environ, start_response, data, status="200 OK"):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response(status, headers)
    return [body]


def application(environ, start_response):
    method = environ["REQUEST_METHOD"]
    path = environ["PATH_INFO"]

    # GET /health
    if method == "GET" and path == "/health":
        return json_response(environ, start_response, {
            "status": "ok", "service": "remote_fund_flow_proxy",
        })

    # GET /fund-flow/{symbol}
    if method == "GET" and path.startswith("/fund-flow/"):
        symbol = path.split("/fund-flow/")[1].strip()
        if not symbol:
            return json_response(environ, start_response,
                {"error": "missing symbol"}, "400 Bad Request")

        result = fetch_one(symbol)
        if result["success"]:
            return json_response(environ, start_response, result)
        else:
            return json_response(environ, start_response, result, "502 Bad Gateway")

    # POST /fund-flow/batch
    if method == "POST" and path == "/fund-flow/batch":
        try:
            body_len = int(environ.get("CONTENT_LENGTH", 0))
            raw = environ["wsgi.input"].read(body_len)
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            return json_response(environ, start_response,
                {"error": "invalid JSON body"}, "400 Bad Request")

        symbols = req.get("symbols", [])
        sleep_seconds = req.get("sleep_seconds", 1.0)
        include_data = req.get("include_data", False)

        results = []
        failed = []
        success_count = 0

        for i, sym in enumerate(symbols):
            if i > 0 and sleep_seconds > 0:
                time.sleep(sleep_seconds)

            result = fetch_one(sym)

            if result["success"]:
                success_count += 1
                entry = {
                    "symbol": result["symbol"],
                    "success": True,
                    "rows": result["rows"],
                    "min_date": result["min_date"],
                    "max_date": result["max_date"],
                }
                if include_data:
                    entry["data"] = result["data"]
                else:
                    entry["data"] = []
                results.append(entry)
            else:
                entry = {"symbol": result["symbol"], "error": result["error"]}
                failed.append(entry)
                results.append({
                    "symbol": result["symbol"],
                    "success": False,
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                    "data": [],
                })

        return json_response(environ, start_response, {
            "success_count": success_count,
            "failed_count": len(failed),
            "results": results,
            "failed": failed,
        })

    # 404
    return json_response(environ, start_response,
        {"error": "not found"}, "404 Not Found")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 18088
    server = make_server(host, port, application)
    logger.info("Starting fund flow proxy on %s:%d", host, port)
    server.serve_forever()
