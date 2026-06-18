
# Appended to vllm/entrypoints/openai/api_server.py at build time — that's why `build_app`
# (referenced at the bottom) is in scope. Wrapped, not patched in place, to survive
# api_server.py line churn across nightly base bumps.

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
