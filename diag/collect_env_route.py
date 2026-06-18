
# Appended to vllm/entrypoints/openai/api_server.py by the image build (see Dockerfile).
# Adds GET /collect_env -> vLLM's collect_env report (secret-scrubbed). PCAI gives no shell and
# no pod logs, so this is the only way to retrieve the env report. `build_app` is wrapped (rather
# than patched in place) so it survives api_server.py line churn across nightly bumps.

def _pcai_collect_env_wrapper(_orig_build_app):
    import re
    from fastapi.responses import PlainTextResponse

    def _scrub(text):
        out = []
        for ln in text.splitlines():
            if re.search(r"(?i)(token|secret|password|api[_-]?key|hf_)", ln) and re.search(r"[=:]\s*\S", ln):
                ln = re.sub(r"([=:]\s*).*", r"\1<redacted>", ln)
            out.append(ln)
        return "\n".join(out)

    def build_app(args):
        app = _orig_build_app(args)

        # sync def (not async): get_pretty_env_info() is blocking (shells out to nvidia-smi /
        # pip list), so let FastAPI run it in a threadpool instead of stalling the event loop.
        @app.get("/collect_env", response_class=PlainTextResponse)
        def _collect_env():
            from vllm.collect_env import get_pretty_env_info
            return _scrub(get_pretty_env_info())

        return app

    return build_app


build_app = _pcai_collect_env_wrapper(build_app)  # noqa: F811
