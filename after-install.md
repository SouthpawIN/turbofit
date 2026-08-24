# After install

`/turbofit` is a plugin command. Hermes Desktop profile sessions (Sirvir included) only load plugins from that profile.

This install now:

1. Enables `turbofit` in every Hermes `config.yaml` (`plugins.enabled`)
2. Links or copies the plugin into each profile's `plugins/turbofit`
3. Publishes the Turbofit skill into each profile's `skills/turbofit`

Then:

1. Fully quit Hermes Desktop (not just the window)
2. Open a new session
3. Run `/turbofit status` then `/turbofit setup`

If a session still says `not a quick/plugin/bundle/skill command: turbofit`, that profile has not reloaded plugins. From a terminal in that profile:

```bash
hermes -p sirvir plugins install --enable https://github.com/SouthpawIN/turbofit.git
```

This is not Windows-specific. Any profile-isolated Desktop or CLI session has the same gap.
