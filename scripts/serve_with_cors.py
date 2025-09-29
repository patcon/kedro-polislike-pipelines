#!/usr/bin/env python3
"""
Script to serve the build directory with CORS headers enabled.

This script starts a simple HTTP server that serves static files from the build
directory with CORS headers to allow cross-origin requests. This is useful for
local development and testing of the static site.

Usage:
    python scripts/serve_with_cors.py [--port PORT] [--directory DIR]
    uv run python scripts/serve_with_cors.py --port 8080
"""

import argparse
import http.server
import os
import socketserver
import sys
from pathlib import Path
from http.server import SimpleHTTPRequestHandler


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers enabled."""

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Override to provide cleaner log messages."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def serve_directory(directory: Path, port: int = 8000) -> None:
    """
    Serve a directory with CORS headers enabled.

    Args:
        directory: Path to the directory to serve
        port: Port number to serve on (default: 8000)
    """
    if not directory.exists():
        print(f"❌ Error: Directory '{directory}' does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"❌ Error: '{directory}' is not a directory")
        sys.exit(1)

    # Change to the target directory
    original_cwd = Path.cwd()
    try:
        os.chdir(directory)

        print(f"🌐 Starting HTTP server with CORS support...")
        print(f"📁 Serving directory: {directory.absolute()}")
        print(f"🚀 Server running at: http://localhost:{port}")
        print(f"🔓 CORS headers enabled for cross-origin requests")
        print(f"Press Ctrl+C to stop the server")
        print("-" * 60)

        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            httpd.serve_forever()

    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Error: Port {port} is already in use")
            print(f"Try using a different port with --port option")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """Main function to parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Serve static files with CORS headers enabled",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/serve_with_cors.py
  python scripts/serve_with_cors.py --port 8080
  python scripts/serve_with_cors.py --directory ./dist --port 3000
        """,
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port number to serve on (default: 8000)"
    )

    parser.add_argument(
        "--directory",
        type=str,
        default="build",
        help="Directory to serve (default: build)",
    )

    args = parser.parse_args()

    # Resolve directory path relative to project root
    project_root = Path(__file__).parent.parent
    directory = project_root / args.directory

    serve_directory(directory, args.port)


if __name__ == "__main__":
    main()
