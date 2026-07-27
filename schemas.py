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
            "profile": {
                "type": "string",
                "description": "Turbofit profile selection: auto or a compatible hardware-*gb profile.",
            },
            "base_url": {
                "type": "string",
                "description": "OpenAI-compatible Turbofit endpoint; defaults to http://127.0.0.1:8091/v1.",
            },
        },
    },
}
