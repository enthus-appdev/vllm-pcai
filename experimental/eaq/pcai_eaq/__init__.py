"""Experimental EAQ qualification plugin; inactive unless explicitly enabled."""
def register():
    import os
    if os.environ.get("PCAI_ENGRAM_DIR"):
        from .runtime import install
        install()
