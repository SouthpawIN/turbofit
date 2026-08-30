"""Schemas exposed by the Turbofit Hermes plugin."""

TURBOFIT_STATUS = {
    "name": "turbofit_status",
    "description": (
        "Inspect the installed Turbofit adaptive local-model provider. Returns "
        "the selected hardware profile, live adaptive rung/routes, gateway health, "
        "and whether Hermes uses Turbofit as its primary or fallback provider."
    ),
    "parameters": {"type": "object", "properties": {}},
}

TURBOFIT_CONFIGURE = {
    "name": "turbofit_configure",
    "description": (
        "Configure Turbofit for Hermes Agent. Registers the named custom provider, "
        "optionally makes model auto the primary model, optionally adds Turbofit to "
        "the canonical fallback provider chain, and can select an automatic or "
        "explicit compatible hardware profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "primary": {
                "type": "boolean",
                "description": "Set custom:turbofit with model auto as Hermes' primary provider.",
            },
            "fallback": {
                "type": "boolean",
                "description": "Add Turbofit to Hermes' fallback_providers chain; false removes it.",
            },
            "fallback_chain": {
                "type": "array",
                "description": "Replace Hermes' ordered fallback chain. Entries contain only provider and model; credentials remain in provider configuration.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["provider", "model"],
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
            },
            "multimodal": {
                "type": "object",
                "description": "Select hardware-fit image, video, music, TTS, and STT integrations.",
                "additionalProperties": False,
                "properties": {
                    "image": {"type": "string"},
                    "video": {"type": "string"},
                    "music": {"type": "string"},
                    "tts": {"type": "string"},
                    "stt": {"type": "string"},
                },
            },
            "profile": {
                "type": "string",
                "description": "Turbofit profile selection: auto or a compatible hardware-*gb profile.",
            },
            "base_url": {
                "type": "string",
                "description": "OpenAI-compatible Turbofit endpoint; defaults to http://127.0.0.1:8091/v1.",
            },
            "publish_tailnet": {
                "type": "boolean",
                "description": "Publish the provider and Hermes dashboard privately with Tailscale Serve and use the resulting HTTPS provider URL.",
            },
            "install_sirvir": {
                "type": "boolean",
                "description": "Install or update the current SouthpawIN/sirvir GitHub profile while preserving its user data.",
            },
            "install_desktop": {
                "type": "boolean",
                "description": "Install or update the bundled native Hermes Desktop Turbofit page.",
            },
            "install_lemonade": {
                "type": "boolean",
                "description": "Install or start the digest-pinned loopback Lemonade Server runtime.",
            },
            "install_native": {
                "type": "boolean",
                "description": "Explicitly install and activate the pinned native runtime. On supported Apple Silicon this downloads the selected MLX snapshot and starts loopback model and gateway processes.",
            },
            "install_freetoken": {
                "type": "boolean",
                "description": "Install pinned FreeToken 0.1.2 as an NVIDIA/CUDA-13 text-only MoE candidate; never auto-promotes without on-box evidence.",
            },
            "dashboard_local_port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "provider_local_port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "dashboard_https_port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "provider_https_port": {"type": "integer", "minimum": 1, "maximum": 65535},
        },
    },
}
