"""The evidence ledger (LEDGER.md). `modelmapper ledger import|build|verify|export|query|pack ...`.

core.py, pack.py and importers/ moved here from src/ledger/ byte-for-byte: PackWriter hashes its importer's source and
core.py into every pack id, so editing them would give every unchanged pack a new id. They still import `ledger.core`
and `config` by bare name and read their paths from the environment; the two lines below make that work: the bare names
resolve to modelmapper.ledger / modelmapper.core.config, and the paths resolved by modelmapper.core.settings are put
into the environment first (same defaults as before).
"""
from modelmapper import legacy as _legacy
from modelmapper.core import settings as _settings

_settings.export_to_environ()
_legacy.install(only={"ledger", "config"})
