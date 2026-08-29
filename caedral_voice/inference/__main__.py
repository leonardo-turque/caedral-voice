"""Enables ``python -m caedral_voice.inference`` -> the CLI entrypoint."""
from caedral_voice.inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
