"""Serve an explicitly selected artifact folder on loopback, with video ranges."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir() or not (directory / "index.html").is_file():
        parser.error("directory must contain index.html")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    from tornado.ioloop import IOLoop
    from tornado.web import Application, StaticFileHandler

    application = Application([(r"/(.*)", StaticFileHandler, dict(path=str(directory), default_filename="index.html"))])
    application.listen(args.port, address="127.0.0.1")
    print(f"Serving {directory} at http://127.0.0.1:{args.port}/", flush=True)
    IOLoop.current().start()


if __name__ == "__main__":
    main()
