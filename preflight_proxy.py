import json, os, sys, uuid, time
from urllib.request import Request, ProxyHandler, build_opener
from urllib.parse import urlencode
from urllib.error import HTTPError

SBUS_URL  = os.getenv("SBUS_URL",  "http://localhost:7000")
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:9000")
_opener = build_opener(ProxyHandler({}))

def _http(method, url, body=None, params=None, headers=None, timeout=10):
    if params: url = f"{url}?{urlencode(params)}"
    h = {"accept": "application/json", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["content-type"] = "application/json"
    req = Request(url, data=data, method=method, headers=h)
    try:
        with _opener.open(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try: return r.status, json.loads(raw) if raw else {}
            except json.JSONDecodeError: return r.status, {"raw": raw}
    except HTTPError as e:
        try: return e.code, json.loads(e.read().decode("utf-8", errors="replace"))
        except Exception: return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}

def check(name, ok, detail=""):
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}{(' — ' + detail) if detail else ''}")
    return ok

def main():
    print("=" * 60)
    print("Preflight: S-Bus + sbus-proxy integration")
    print("=" * 60)

    all_ok = True

    st, body = _http("GET", f"{SBUS_URL}/admin/delivery-log")
    if st == 200:
        all_ok &= check("S-Bus reachable and admin mode enabled", True)
    elif st == 403:
        all_ok &= check("S-Bus reachable and admin mode enabled", False,
                        "got 403 — start server with SBUS_ADMIN_ENABLED=1")
    else:
        all_ok &= check("S-Bus reachable and admin mode enabled", False,
                        f"got HTTP {st} from /admin/delivery-log (server down?)")

    test_agent = f"preflight_{uuid.uuid4().hex[:6]}"
    suffix = f"_{uuid.uuid4().hex[:6]}"
    shard_key = f"preflight_shard{suffix}"
    _http("POST", f"{SBUS_URL}/shard", {
        "key": shard_key, "content": "init", "goal_tag": "preflight",
    })
    st, body = _http("POST", f"{SBUS_URL}/delivery_log/register", {
        "agent_id":    test_agent,
        "session_id":  "preflight",
        "shards_used": [shard_key],
        "source":      "preflight",
    })
    all_ok &= check("S-Bus /delivery_log/register responds", st == 200, f"got {st}")
    if st == 200 and isinstance(body, dict):
        all_ok &= check("/delivery_log/register recorded the entry",
                        body.get("registered", 0) >= 1,
                        f"response: {body}")

    st, _ = _http("GET", f"{PROXY_URL}/health")
    all_ok &= check("Proxy /health returns 200", st == 200, f"got {st}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        check("End-to-end proxy round-trip (OPENAI_API_KEY not set)", True,
              "skipped — set OPENAI_API_KEY for full preflight")
    else:
        vocab_hit = os.environ.get("PREFLIGHT_VOCAB_HIT", "models_state")
        probe_shard = f"{vocab_hit}{suffix}"
        _http("POST", f"{SBUS_URL}/shard", {
            "key": probe_shard, "content": "init", "goal_tag": "preflight",
        })

        probe_body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content":
                 f"You have access to {vocab_hit}. Output one word: OK."},
                {"role": "user", "content": "Say OK."}
            ],
            "temperature": 0.0, "max_tokens": 5,
        }
        req = Request(
            f"{PROXY_URL}/v1/chat/completions",
            data=json.dumps(probe_body).encode(),
            method="POST",
            headers={
                "content-type":          "application/json",
                "authorization":         f"Bearer {api_key}",
                "X-SBus-Agent-Id":       test_agent,
                "X-SBus-Session-Id":     "preflight",
                "X-SBus-Shard-Suffix":   suffix,
            },
        )
        t0 = time.time()
        try:
            with _opener.open(req, timeout=30) as r:
                st = r.status
                body = r.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            st   = e.code
            body = e.read().decode("utf-8", errors="replace")
        except Exception as e:
            st   = 0
            body = str(e)
        dur = int((time.time() - t0) * 1000)
        check("Proxy forwarded to OpenAI and returned 2xx",
              200 <= st < 300, f"got {st} in {dur}ms")

        time.sleep(0.3)
        st, dump = _http("GET", f"{SBUS_URL}/admin/delivery-log")

        entries = {}
        if st == 200 and isinstance(dump, dict):
            entries = dump.get("agents", {}).get(test_agent, {})
        found_suffixed = probe_shard in (entries or {})
        all_ok &= check(
            f"Proxy registered '{probe_shard}' under agent {test_agent}",
            found_suffixed,
            f"keys found: {list((entries or {}).keys())}"
        )

    print("=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — safe to run exp_proxy_ph2.py")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED — fix before running the big experiment")
        sys.exit(1)

if __name__ == "__main__":
    main()
