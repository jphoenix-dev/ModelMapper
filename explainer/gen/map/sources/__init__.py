"""One module per source of findings. Each exposes SOURCE (who/what/how, shown as provenance on the map) and
load(ctx), which calls ctx.add_map(...) for concept maps and ctx.add_finding(...) for claims about units.
Register a new module in build_map.SOURCES."""
