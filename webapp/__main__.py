"""Start the live lab:  python -m webapp  [--port 8000] [--no-browser]"""
import argparse
import threading
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Glass-Box Insight live lab")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    print(f"Glass-Box Insight lab running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("webapp.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
