"""SharePoint document-library intake (architecture doc agent #1, server-side).

App-only authentication via Azure ACS (the legacy SharePoint app-only flow your
host uses — AD_TOKEN_URL = accounts.accesscontrol.windows.net). Implemented with
stdlib urllib so it adds NO dependency. Lists + downloads files from a library and
returns them in the same shape Submission Intake expects:
    [{"filename": ..., "content_bytes": ..., "source": "sharepoint:<url>"}]

Config-driven (nebag_SHAREPOINT_* or host SH_*/SHAREPOINT_* names). Every call is
best-effort: on auth/network failure it returns [] with a clear log, never crashing
the pipeline. Tokens/secrets are never logged.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .config import Settings
from .observability import get_logger

_SP_PRINCIPAL = "00000003-0000-0ff1-ce00-000000000000"  # SharePoint Online resource id


class SharePointClient:
    def __init__(self, settings: Settings):
        self._s = settings
        self._log = get_logger("sharepoint")

    def is_configured(self) -> bool:
        s = self._s
        return bool(
            s.sharepoint_site_url and s.sharepoint_client_id and s.sharepoint_client_secret
            and s.sharepoint_tenant_id and s.sharepoint_token_url
        )

    def _site_host(self) -> str:
        return (self._s.sharepoint_site_url or "").split("//")[-1].split("/")[0]

    def _site_base(self) -> str:
        s = self._s
        base = (s.sharepoint_site_url or "").rstrip("/")
        return f"{base}/sites/{s.sharepoint_site_name}" if s.sharepoint_site_name else base

    def _folder_server_relative(self) -> str:
        s = self._s
        prefix = f"/sites/{s.sharepoint_site_name}" if s.sharepoint_site_name else ""
        return f"{prefix}/{s.sharepoint_library_name}".replace("//", "/")

    def _acs_token(self) -> str:
        s = self._s
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": f"{s.sharepoint_client_id}@{s.sharepoint_tenant_id}",
            "client_secret": s.sharepoint_client_secret,
            "resource": f"{_SP_PRINCIPAL}/{self._site_host()}@{s.sharepoint_tenant_id}",
        }).encode("utf-8")
        req = urllib.request.Request(
            s.sharepoint_token_url, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(s.llm_timeout_seconds)) as resp:
            return json.loads(resp.read().decode("utf-8"))["access_token"]

    def _get(self, url: str, token: str, raw: bool = False):
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}",
                          "Accept": "application/json;odata=nometadata"},
        )
        with urllib.request.urlopen(req, timeout=float(self._s.llm_timeout_seconds)) as resp:
            return resp.read() if raw else json.loads(resp.read().decode("utf-8"))

    def list_files(self, token: Optional[str] = None) -> List[Dict[str, Any]]:
        token = token or self._acs_token()
        folder = urllib.parse.quote(self._folder_server_relative())
        url = f"{self._site_base()}/_api/web/GetFolderByServerRelativeUrl('{folder}')/Files"
        return self._get(url, token).get("value", [])

    def download(self, server_relative_url: str, token: str) -> bytes:
        path = urllib.parse.quote(server_relative_url)
        url = f"{self._site_base()}/_api/web/GetFileByServerRelativeUrl('{path}')/$value"
        return self._get(url, token, raw=True)

    def pull_files(self) -> List[Dict[str, Any]]:
        """List + download every file in the configured library. Resilient."""
        if not self.is_configured():
            return []
        try:
            token = self._acs_token()
            entries = self.list_files(token)
        except Exception as exc:
            self._log.warning("sharepoint list failed: %s: %s", type(exc).__name__, exc)
            return []
        out: List[Dict[str, Any]] = []
        for f in entries:
            srv = f.get("ServerRelativeUrl")
            if not srv:
                continue
            try:
                data = self.download(srv, token)
            except Exception as exc:
                self._log.warning("sharepoint download failed for %s: %s", f.get("Name"), exc)
                continue
            out.append({"filename": f.get("Name"), "content_bytes": data, "source": f"sharepoint:{srv}"})
        self._log.info("sharepoint: pulled %d file(s) from %s", len(out), self._folder_server_relative())
        return out
