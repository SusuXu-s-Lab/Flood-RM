from collect_sources.mgr import bind_module

bind_module(__name__, "collect_sources.mgr.hurdat2", globals())

for _name in ("collect_hurdat2", "parse", "parse_hurdat2"):
    if _name in globals():
        globals()[_name].__module__ = __name__
