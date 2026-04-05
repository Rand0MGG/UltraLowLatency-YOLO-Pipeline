from __future__ import annotations

import argparse
import webbrowser

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cvcap local control panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    if not args.no_open:
        webbrowser.open(f"http://{args.host}:{args.port}", new=2)

    uvicorn.run("cvcap.web.server:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
