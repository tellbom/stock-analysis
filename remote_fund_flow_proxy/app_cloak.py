"""
Eastmoney fund flow proxy — CloakBrowser edition.

Uses stealth Chromium (BoringSSL) instead of Python OpenSSL/urllib3
to test whether browser-based TLS eliminates random disconnections.

Start: python app.py
"""
import json
import logging
import time
import sys
from wsgiref.simple_server import make_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fund_flow_proxy_cloak")

EM_BASE = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
FIELDS1 = "f1,f2,f3,f7"
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
LMT = 100000
KLT = 101

FIELD_NAMES = [
    "trade_date", "main_net", "small_net", "medium_net",
    "large_net", "super_net", "main_net_rate", "small_net_rate",
    "medium_net_rate", "large_net_rate", "super_net_rate",
    "close", "pct_change",
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

# Lazy-init browser (expensive — shared across requests)
_browser = None


def get_browser():
    global _browser
    if _browser is None:
        from cloakbrowser import launch
        logger.info("Launching stealth Chromium (one-time)...")
        _browser = launch(headless=True)
        logger.info("Chromium ready")
    return _browser


def resolve_market(symbol):
    symbol = symbol.strip().upper()
    if "." in symbol:
        code, suffix = symbol.split(".", 1)
        m = {"SH": 1, "1": 1, "SZ": 0, "0": 0, "BJ": 2, "2": 2}
        if suffix in m:
            return code, m[suffix]
        raise ValueError("Unknown suffix: %s" % suffix)
    for prefixes, mkt in MARKET_PREFIXES:
        for pfx in prefixes:
            if symbol.startswith(pfx):
                return symbol, mkt
    raise ValueError("Unknown symbol: %s" % symbol)


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
    """Fetch via CloakBrowser — navigates to API URL, extracts JSON from page."""
    code, mkt = resolve_market(symbol)
    secid = "%s.%s" % (mkt, code)
    url = "%s?lmt=%s&klt=%s&secid=%s&fields1=%s&fields2=%s" % (
        EM_BASE, LMT, KLT, secid, FIELDS1, FIELDS2
    )

    browser = get_browser()
    page = browser.new_page()

    try:
        t0 = time.time()
        resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        status = resp.status if resp else 0
        elapsed = time.time() - t0
        logger.info("symbol=%s secid=%s HTTP=%d time=%.1fs", symbol, secid, status, elapsed)

        if status != 200:
            return {
                "symbol": code, "success": False, "rows": 0,
                "min_date": None, "max_date": None,
                "source": "eastmoney_push2his_cloakbrowser",
                "data": [], "error": "HTTP %d" % status,
            }

        # Extract JSON from the page body (browser renders raw JSON as text)
        body = page.evaluate("document.body.innerText")
        data = json.loads(body)

        if data.get("rc") != 0:
            return {
                "symbol": code, "success": False, "rows": 0,
                "min_date": None, "max_date": None,
                "source": "eastmoney_push2his_cloakbrowser",
                "data": [], "error": "API rc=%s" % data.get("rc"),
            }

        klines = data.get("data", {}).get("klines")
        if not klines:
            return {
                "symbol": code, "success": True, "rows": 0,
                "min_date": None, "max_date": None,
                "source": "eastmoney_push2his_cloakbrowser",
                "data": [], "error": None,
            }

        records = [parse_kline(k, code) for k in klines]
        min_date = records[0]["trade_date"]
        max_date = records[-1]["trade_date"]
        n = len(records)

        logger.info("symbol=%s success=True rows=%d min=%s max=%s",
                    code, n, min_date, max_date)

        return {
            "symbol": code, "success": True, "rows": n,
            "min_date": min_date, "max_date": max_date,
            "source": "eastmoney_push2his_cloakbrowser",
            "data": records, "error": None,
        }

    except Exception as e:
        logger.error("symbol=%s error=%s: %s", symbol, type(e).__name__, e)
        return {
            "symbol": code, "success": False, "rows": 0,
            "min_date": None, "max_date": None,
            "source": "eastmoney_push2his_cloakbrowser",
            "data": [], "error": "%s: %s" % (type(e).__name__, e),
        }
    finally:
        page.close()


# ---------------------------------------------------------------------------
# WSGI app
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

    if method == "GET" and path == "/health":
        return json_response(environ, start_response, {
            "status": "ok", "service": "remote_fund_flow_proxy_cloak",
        })

    if method == "GET" and path.startswith("/fund-flow/"):
        symbol = path.split("/fund-flow/")[1].strip()
        if not symbol:
            return json_response(environ, start_response,
                {"error": "missing symbol"}, "400 Bad Request")
        result = fetch_one(symbol)
        status = "200 OK" if result["success"] else "502 Bad Gateway"
        return json_response(environ, start_response, result, status)

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
                    "symbol": result["symbol"], "success": True,
                    "rows": result["rows"],
                    "min_date": result["min_date"], "max_date": result["max_date"],
                }
                entry["data"] = result["data"] if include_data else []
                results.append(entry)
            else:
                entry = {"symbol": result["symbol"], "error": result["error"]}
                failed.append(entry)
                results.append({
                    "symbol": result["symbol"], "success": False, "rows": 0,
                    "min_date": None, "max_date": None, "data": [],
                })

        return json_response(environ, start_response, {
            "success_count": success_count,
            "failed_count": len(failed),
            "results": results,
            "failed": failed,
        })

    return json_response(environ, start_response,
        {"error": "not found"}, "404 Not Found")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 18089
    server = make_server(host, port, application)
    logger.info("CloakBrowser proxy on %s:%d", host, port)
    server.serve_forever()
