from __future__ import annotations

"""
Entry point wrapper for the pygame_worm app.

The full application logic lives in `pygame_worm/app.py`.
This small module keeps the public entry stable (`python -m pygame_worm.main`)
while making the codebase more modular and maintainable.
"""

from .app import run


if __name__ == "__main__":
    run()

