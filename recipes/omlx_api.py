"""oMLX 本地 API 小工具：密钥从 settings.json 读取，绝不打印。"""
import json, os, time, urllib.request
BASE = "http://127.0.0.1:8000"
_KEY = json.load(open(os.path.expanduser("~/.omlx/settings.json")))["auth"]["api_key"]
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def call(path, body=None, method=None, timeout=900, base=BASE, key=True):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {_KEY}")
    t = time.time()
    with _opener.open(req, timeout=timeout) as r:
        out = r.read()
    try:
        return json.loads(out), time.time() - t
    except Exception:
        return out.decode(errors="replace"), time.time() - t

def _admin_cookie():
    import sys
    sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages")
    from itsdangerous import URLSafeTimedSerializer
    sk = json.load(open(os.path.expanduser("~/.omlx/settings.json")))["auth"]["secret_key"]
    return "omlx_admin_session=" + URLSafeTimedSerializer(sk).dumps({"admin": True, "remember": False})

def admin(path, body=None, method=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + "/admin" + path, data=data, method=method or ("POST" if data is not None else "GET"))
    req.add_header("Content-Type", "application/json")
    req.add_header("Cookie", _admin_cookie())
    with _opener.open(req, timeout=timeout) as r:
        out = r.read()
    try:
        return json.loads(out)
    except Exception:
        return out.decode(errors="replace")
