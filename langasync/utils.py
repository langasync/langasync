import os
import uuid


def generate_uuid() -> str:
    frozen = os.environ.get("FROZEN_UUID")
    if frozen:
        return frozen
    return str(uuid.uuid4())
