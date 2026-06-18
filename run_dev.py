"""Local dev-server launcher (Windows-safe).

Why this exists instead of `uvicorn app.main:app --reload`:

uvicorn picks the asyncio event loop with this factory (uvicorn/loops/asyncio.py):

    if sys.platform == "win32" and not use_subprocess:
        return asyncio.ProactorEventLoop      # single process
    return asyncio.SelectorEventLoop          # --reload or --workers>1

`--reload` runs the server in a subprocess (use_subprocess=True), which forces the
SelectorEventLoop.  On Windows select() is hard-capped at 512 file descriptors, and
the concurrent search/export pipeline opens enough sockets to exceed that, crashing
the worker with "ValueError: too many file descriptors in select()".  The reloader
parent then keeps port 8000 bound with no live worker, so every request hangs.

Running single-process (reload=False) gives the ProactorEventLoop (Windows IOCP),
which has no such FD limit.  A crash also frees the port cleanly instead of leaving
an orphaned worker behind.

Trade-off: no auto-reload.  Restart this script after code changes.

Usage:
    .venv\\Scripts\\python.exe run_dev.py
"""
from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        loop="asyncio",   # on Windows single-process this resolves to ProactorEventLoop
        reload=False,     # reload would force the 512-FD-limited SelectorEventLoop
    )
