"""End-to-end demo with NO server and NO credentials.

Runs the full Start->End workflow on a mock submission and prints both the
internal result and the One AI envelope the chatbot would receive.

    python run_demo.py
"""

from __future__ import annotations

import json

from app.config import Settings
from app.contracts import IntakeMode, nebagInput, to_oneai_envelope
from app.service import nebagAgent


def main():
    # Inject mock-LLM settings so it runs with zero credentials.
    settings = Settings(llm_provider="mock")
    agent = nebagAgent(settings=settings)

    inp = nebagInput(
        query="Process this new D&O submission from Acme Health Systems.",
        mode=IntakeMode.UPLOAD,
        files=[{"filename": "application.pdf", "content_type": "application/pdf"}],
        app_user_id="demo-user",
    )

    result = agent.run(inp)

    print("=== Internal nebagResult ===")
    print(result.model_dump_json(indent=2))

    print("\n=== Status trail (node-by-node) ===")
    for line in result.status_trail:
        print("  -", line)

    print("\n=== One AI envelope (what the chatbot receives) ===")
    print(json.dumps(to_oneai_envelope(result, as_table=True), indent=2))

    if result.review_required:
        print("\n=== Simulating HITL resume (human approves employee_count) ===")
        resumed = agent.resume(
            result.submission_id,
            corrections=[{"name": "employee_count", "value": 1200}],
        )
        print("review_required after resume:", resumed.review_required)
        print("summary:", resumed.summary)


if __name__ == "__main__":
    main()
