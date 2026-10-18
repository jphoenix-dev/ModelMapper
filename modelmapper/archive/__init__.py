"""The 16-17 Oct 2026 one-off analyses, moved here byte-for-byte from src/ and kept for the record.

They still import their neighbours by bare name (`from config import env`) and take Path(__file__).parents[1] as the
repo root, so do not import them as modelmapper.archive.<name>. Run them with `modelmapper archive <name> [args]`
(or `modelmapper run src/<name>.py [args]`), which resolves those names to the new modules and gives the script its
old __file__, so its outputs land where they always did (modelmapper/legacy.py).
"""
