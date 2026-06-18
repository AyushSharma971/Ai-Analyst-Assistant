"""Register the nebag agent with the One AI core backend (or just print payload).

    # Print the payload (no network):
    python scripts/register_agent.py

    # Actually register (needs nebag_ONEAI_REGISTER_URL or --url):
    python scripts/register_agent.py --post --url https://<core-backend>/agents \
        --backend-url https://<this-service-host>

Registering in the One AI core backend surfaces the agent in the Marketplace
with no frontend deploy (design record §6).
"""

from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.registration import build_registration_payload, register_agent


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/POST the One AI registration payload.")
    ap.add_argument("--post", action="store_true", help="POST to the core backend")
    ap.add_argument("--url", help="core-backend registration URL (else nebag_ONEAI_REGISTER_URL)")
    ap.add_argument("--backend-url", help="public URL of THIS service ({base}/{slug})")
    args = ap.parse_args()

    settings = Settings()
    payload = build_registration_payload(settings, base_url=args.backend_url)

    print(json.dumps(payload, indent=2))

    if args.post:
        result = register_agent(settings, payload=payload, register_url=args.url)
        print("\n=== Registration response ===")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
