"""
Standalone connectivity test server for the client VM (no Docker, no app code).

Copy this file next to a .env and run:
    python test_server.py            # listens on 0.0.0.0:5000
    python test_server.py 8080       # or another port

Endpoints (GET):
    /                         -> list of endpoints
    /health                   -> plain liveness, use as the APIM backend health probe
    /echo                     -> returns path, query and headers as received (verifies APIM forwarding)
    /connect_postgresql_check -> DNS + TCP + login + SELECT on DB_HOST
    /connect_ai_search_check  -> DNS + TCP + document count on AZURE_SEARCH_INDEX
    /AI_foundary_check        -> DNS + TCP + 1 tiny Claude message on AZURE_AI_FOUNDRY_ENDPOINT
    /azure_openai_check       -> DNS + TCP + 1 tiny chat + 1 embedding on AZURE_OPENAI_ENDPOINT
    /all                      -> runs every check above

Every check reports each layer separately so a failure says WHERE it broke:
    dns  -> what the hostname resolved to (10.x / 172.16-31.x / 192.168.x = private endpoint)
    tcp  -> whether the port opened (NSG / firewall / private endpoint routing)
    call -> real auth + request (keys, firewall allow-lists, deployment names)

Only stdlib is required. Postgres needs one of: psycopg (v3), psycopg2, or asyncpg;
if none is installed, the Postgres check stops after the TCP layer.
"""

import ipaddress
import json
import os
import socket
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

TIMEOUT = 15


# --------------------------------------------------------------------------- .env

def load_env(path=".env"):
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=True)
        return "python-dotenv"
    except ImportError:
        pass
    if not os.path.exists(path):
        return "no .env found"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.split(" #")[0].strip().strip('"').strip("'")
            os.environ[key.strip()] = value
    return "builtin parser"


def env(name, default=""):
    return os.getenv(name, default).strip().strip('"').strip("'")


def mask(value):
    if not value:
        return "<EMPTY>"
    return value[:4] + "..." + value[-2:] if len(value) > 8 else "***"


# ------------------------------------------------------------------ layer checks

def dns_check(host):
    start = time.time()
    try:
        ips = sorted({ai[4][0] for ai in socket.getaddrinfo(host, None)})
        private = [ip for ip in ips if ipaddress.ip_address(ip).is_private]
        return {
            "ok": True,
            "host": host,
            "ips": ips,
            "resolves_private": bool(private),
            "hint": "private endpoint" if private else "public IP (no private endpoint / private DNS zone not linked?)",
            "ms": round((time.time() - start) * 1000),
        }
    except Exception as e:
        return {"ok": False, "host": host, "error": f"{type(e).__name__}: {e}",
                "hint": "DNS failed - check private DNS zone link / VM DNS server"}


def tcp_check(host, port):
    start = time.time()
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            pass
        return {"ok": True, "port": port, "ms": round((time.time() - start) * 1000)}
    except Exception as e:
        return {"ok": False, "port": port, "error": f"{type(e).__name__}: {e}",
                "hint": "port blocked - NSG / firewall / service network rules"}


def http_request(url, method="GET", headers=None, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ssl.create_default_context()) as resp:
            text = resp.read().decode("utf-8", "replace")
            return {"ok": True, "status": resp.status, "ms": round((time.time() - start) * 1000),
                    "body": _try_json(text)}
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        return {"ok": False, "status": e.code, "ms": round((time.time() - start) * 1000),
                "body": _try_json(text)[:1500] if isinstance(_try_json(text), str) else _try_json(text),
                "hint": _http_hint(e.code)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "hint": "SSL error -> proxy/SSL inspection; set SSL_CERT_FILE to the corp CA bundle"
                if "SSL" in type(e).__name__ or "CERTIFICATE" in str(e).upper() else None}


def _try_json(text):
    try:
        return json.loads(text)
    except Exception:
        return text[:1500]


def _http_hint(code):
    return {
        401: "bad/missing key",
        403: "key OK but blocked - service firewall / public network access disabled / IP not allowed",
        404: "wrong index / deployment name / api path",
        429: "reachable, but throttled (quota)",
    }.get(code)


def run_layers(host, port, call):
    result = {"dns": dns_check(host)}
    if not result["dns"]["ok"]:
        return _finish(result)
    result["tcp"] = tcp_check(host, port)
    if not result["tcp"]["ok"]:
        return _finish(result)
    try:
        result["call"] = call()
    except Exception as e:
        result["call"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc().splitlines()[-3:]}
    return _finish(result)


def _finish(result):
    result["ok"] = all(layer.get("ok") for layer in result.values() if isinstance(layer, dict))
    return result


# ---------------------------------------------------------------------- services

def check_postgres():
    host = env("DB_HOST")
    port = int(env("DB_PORT", "5432") or 5432)
    user, password = env("DB_USER"), env("DB_PASSWORD")
    databases = [d for d in {env("APP_DATABASE"), env("AUTH_DATABASE")} if d] or ["postgres"]
    config = {"DB_HOST": host, "DB_PORT": port, "DB_USER": user, "DB_PASSWORD": mask(password),
              "databases": databases}
    if not host:
        return {"ok": False, "config": config, "error": "DB_HOST is empty"}

    def call():
        out = {"ok": True, "databases": {}}
        for db in databases:
            r = _pg_query(host, port, user, password, db)
            out["databases"][db] = r
            out["ok"] = out["ok"] and r.get("ok", False)
        return out

    return {"config": config, **run_layers(host, port, call)}


def _pg_query(host, port, user, password, db):
    sql = "SELECT version(), current_user, current_database()"
    start = time.time()
    try:
        try:
            import psycopg
            with psycopg.connect(host=host, port=port, user=user, password=password, dbname=db,
                                 sslmode="require", connect_timeout=TIMEOUT) as conn:
                row = conn.execute(sql).fetchone()
            driver = "psycopg3"
        except ImportError:
            try:
                import psycopg2
                conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=db,
                                        sslmode="require", connect_timeout=TIMEOUT)
                cur = conn.cursor()
                cur.execute(sql)
                row = cur.fetchone()
                conn.close()
                driver = "psycopg2"
            except ImportError:
                import asyncio
                import asyncpg

                async def q():
                    c = await asyncpg.connect(host=host, port=port, user=user, password=password,
                                              database=db, ssl="require", timeout=TIMEOUT)
                    try:
                        return await c.fetchrow(sql)
                    finally:
                        await c.close()
                row = tuple(asyncio.run(q()))
                driver = "asyncpg"
        return {"ok": True, "driver": driver, "version": row[0][:60], "user": row[1], "db": row[2],
                "ms": round((time.time() - start) * 1000)}
    except ImportError:
        return {"ok": False, "error": "no Postgres driver installed",
                "hint": "pip install 'psycopg[binary]'  (TCP layer above already proves reachability)"}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        hint = None
        if "password authentication failed" in msg:
            hint = "reachable; wrong DB_USER/DB_PASSWORD"
        elif "no pg_hba.conf entry" in msg:
            hint = "reachable; server firewall rule / pg_hba does not allow this VM IP"
        elif "does not exist" in msg:
            hint = "reachable and logged in; database name wrong"
        return {"ok": False, "error": msg, "hint": hint}


def check_ai_search():
    endpoint = env("AZURE_SEARCH_SERVICE_ENDPOINT").rstrip("/")
    key = env("AZURE_SEARCH_ADMIN_KEY")
    indexes = [i for i in (env("AZURE_SEARCH_INDEX"), env("AZURE_SEARCH_CODE_INDEX")) if i]
    config = {"AZURE_SEARCH_SERVICE_ENDPOINT": endpoint, "AZURE_SEARCH_ADMIN_KEY": mask(key), "indexes": indexes}
    if not endpoint:
        return {"ok": False, "config": config, "error": "AZURE_SEARCH_SERVICE_ENDPOINT is empty"}
    host = urlparse(endpoint).hostname

    def call():
        out = {"ok": True, "indexes": {}}
        for idx in indexes or [None]:
            url = (f"{endpoint}/indexes/{idx}/docs/$count?api-version=2024-07-01" if idx
                   else f"{endpoint}/indexes?api-version=2024-07-01&$select=name")
            r = http_request(url, headers={"api-key": key})
            out["indexes"][idx or "<list>"] = r
            out["ok"] = out["ok"] and r["ok"]
        return out

    return {"config": config, **run_layers(host, 443, call)}


def check_ai_foundry():
    endpoint = env("AZURE_AI_FOUNDRY_ENDPOINT").rstrip("/")
    key = env("AZURE_AI_FOUNDRY_KEY")
    deployment = env("AZURE_AI_FOUNDRY_CLAUDE_DEPLOYMENT", "claude-sonnet-4-5")
    config = {"AZURE_AI_FOUNDRY_ENDPOINT": endpoint, "AZURE_AI_FOUNDRY_KEY": mask(key),
              "AZURE_AI_FOUNDRY_CLAUDE_DEPLOYMENT": deployment}
    if not endpoint:
        return {"ok": False, "config": config, "error": "AZURE_AI_FOUNDRY_ENDPOINT is empty"}
    host = urlparse(endpoint).hostname

    def call():
        # Endpoint is ".../anthropic" -> Anthropic Messages API at /v1/messages
        r = http_request(
            f"{endpoint}/v1/messages", method="POST",
            headers={"x-api-key": key, "api-key": key, "anthropic-version": "2023-06-01"},
            body={"model": deployment, "max_tokens": 16,
                  "messages": [{"role": "user", "content": "Reply with the single word: pong"}]},
        )
        if r["ok"] and isinstance(r["body"], dict):
            text = "".join(b.get("text", "") for b in r["body"].get("content", []))
            r["body"] = {"reply": text, "model": r["body"].get("model"), "usage": r["body"].get("usage")}
        return r

    return {"config": config, **run_layers(host, 443, call)}


def check_azure_openai():
    endpoint = env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    key = env("AZURE_OPENAI_KEY")
    chat = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    chat_ver = env("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    emb = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    emb_ver = env("AZURE_OPENAI_EMBEDDING_API_VERSION", chat_ver)
    config = {"AZURE_OPENAI_ENDPOINT": endpoint, "AZURE_OPENAI_KEY": mask(key),
              "chat_deployment": chat, "embedding_deployment": emb}
    if not endpoint:
        return {"ok": False, "config": config, "error": "AZURE_OPENAI_ENDPOINT is empty"}
    host = urlparse(endpoint).hostname

    def call():
        out = {"ok": True}
        if chat:
            r = http_request(f"{endpoint}/openai/deployments/{chat}/chat/completions?api-version={chat_ver}",
                             method="POST", headers={"api-key": key},
                             body={"messages": [{"role": "user", "content": "Reply: pong"}], "max_tokens": 5})
            if r["ok"] and isinstance(r["body"], dict):
                r["body"] = {"reply": r["body"]["choices"][0]["message"]["content"]}
            out["chat"] = r
            out["ok"] = out["ok"] and r["ok"]
        if emb:
            r = http_request(f"{endpoint}/openai/deployments/{emb}/embeddings?api-version={emb_ver}",
                             method="POST", headers={"api-key": key}, body={"input": "ping"})
            if r["ok"] and isinstance(r["body"], dict):
                r["body"] = {"dimensions": len(r["body"]["data"][0]["embedding"])}
            out["embedding"] = r
            out["ok"] = out["ok"] and r["ok"]
        return out

    return {"config": config, **run_layers(host, 443, call)}


CHECKS = {
    "/connect_postgresql_check": check_postgres,
    "/connect_ai_search_check": check_ai_search,
    "/ai_foundary_check": check_ai_foundry,
    "/ai_foundry_check": check_ai_foundry,
    "/azure_openai_check": check_azure_openai,
}


# ------------------------------------------------------------------------ server

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        # Tolerate an APIM API URL suffix being forwarded (e.g. /vmtest/health -> /health)
        path = "/" + parsed.path.rstrip("/").split("/")[-1].lower() if parsed.path.strip("/") else "/"

        if path == "/":
            return self._send(200, {"server": "vm connectivity test", "hostname": socket.gethostname(),
                                    "endpoints": ["/health", "/echo", *CHECKS, "/all"]})
        if path == "/health":
            return self._send(200, {"status": "ok", "hostname": socket.gethostname(), "time": time.time()})
        if path == "/echo":
            return self._send(200, {"method": "GET", "raw_path": self.path, "client": self.client_address[0],
                                    "headers": dict(self.headers)})
        if path == "/all":
            results = {name: fn() for name, fn in CHECKS.items() if name != "/ai_foundry_check"}
            return self._send(200, {"ok": all(r.get("ok") for r in results.values()), "results": results})
        if path in CHECKS:
            result = CHECKS[path]()
            return self._send(200 if result.get("ok") else 502, result)
        return self._send(404, {"error": f"unknown path {parsed.path}", "endpoints": list(CHECKS)})

    do_POST = do_GET

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    loader = load_env(env_path if os.path.exists(env_path) else ".env")
    print(f".env loaded via: {loader}")
    for proxy_var in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        if os.getenv(proxy_var) or os.getenv(proxy_var.lower()):
            print(f"note: {proxy_var} is set -> HTTPS checks go through the proxy")
    print(f"Listening on 0.0.0.0:{port}  (try: curl http://localhost:{port}/all)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
