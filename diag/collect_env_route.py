
# Appended to the active ASGI factory in vllm/entrypoints/launchers/app.py.
# Wrapping preserves upstream app initialization across base-image updates.

def _pcai_collect_env_wrapper(_orig_build_app):
    from fastapi.responses import PlainTextResponse

    # *args/**kwargs: match whatever signature vLLM's build_app has — it varies across versions
    # (here it's build_app(args, supported_tasks, model_config)), so never pin it.
    def build_app(*args, **kwargs):
        app = _orig_build_app(*args, **kwargs)

        # sync def (not async): get_pretty_env_info() is blocking (shells out to nvidia-smi /
        # pip list), so let FastAPI run it in a threadpool instead of stalling the event loop.
        @app.get("/collect_env", response_class=PlainTextResponse)
        def _collect_env():
            from vllm.collect_env import get_pretty_env_info
            return get_pretty_env_info()

        return app

    return build_app


build_app = _pcai_collect_env_wrapper(build_app)  # noqa: F811
