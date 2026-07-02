"""SharePoint intake probe — no secrets/filenames printed (count + status only)."""

from __future__ import annotations

import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings, _env_overlay
from app.sharepoint import SharePointClient


def main():
    s = Settings(**_env_overlay())
    c = SharePointClient(s)
    print("configured:", c.is_configured())
    print("site host:", c._site_host(), "| site base:", c._site_base())
    print("library folder (server-relative):", c._folder_server_relative())
    if not c.is_configured():
        return

    print("\n--- ACS app-only token ---")
    token = None
    try:
        token = c._acs_token()
        print("token: OK (acquired, not shown)")
    except urllib.error.HTTPError as e:
        print("token FAILED HTTP", e.code, "->", e.read().decode("utf-8", "replace")[:120].replace("\n", " "))
    except Exception as e:
        print("token FAILED:", type(e).__name__, "-", str(e)[:140])

    if token:
        print("\n--- list library files ---")
        try:
            files = c.list_files(token)
            print("list OK -> file count:", len(files))
        except urllib.error.HTTPError as e:
            print("list FAILED HTTP", e.code, "->", e.read().decode("utf-8", "replace")[:160].replace("\n", " "))
        except Exception as e:
            print("list FAILED:", type(e).__name__, "-", str(e)[:160])


if __name__ == "__main__":
    main()
