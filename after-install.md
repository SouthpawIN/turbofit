# After install

`/turbofit` is a **plugin command**. It only exists in a session that has *loaded* the plugin, and
plugins load at **startup**. So right after a fresh install, `/turbofit` is not recognized until you
fully quit and reopen. If a session says:

```
not a quick/plugin/bundle/skill command: turbofit
```

that profile has not loaded the plugin — either the install never actually ran, or the app was not
restarted. This doc is the fix for that. (It is not Windows-specific; any Desktop or CLI profile has
the same behavior.)

## 1. Install it for real

Pasting *"Install TurboFit from https://github.com/SouthpawIN/turbofit.git"* only works when the
agent running it decides to **execute** it. Weaker agents sometimes just describe the steps back
instead of running them — so nothing gets installed and `/turbofit` never appears. To install without
depending on that, run the plugin command directly (this is the reliable path):

```bash
hermes plugins install --enable https://github.com/SouthpawIN/turbofit.git
```

That installs **one package**: it enables `turbofit` in this home's `config.yaml` (`plugins.enabled`),
copies the plugin into `plugins/turbofit`, and brings TurboSouth and the `s0uthpaw` pet along with it.

## 2. Verify the install actually landed

Check **before** you expect `/turbofit` to work — this catches a no-op install immediately:

```bash
hermes plugins info turbofit     # must print: Status: enabled
hermes plugins list              # the turbofit row must say "enabled"
```

If `hermes plugins info turbofit` says the plugin is unknown / not installed, the install did not
complete — re-run step 1. (When the plugin is not installed, `/turbofit` also returns
`not a … command`.)

## 3. Reload plugins (required)

Fully quit Hermes Desktop (**⌘Q**, not just close the window) and open a new session. Closing the
window or opening a new chat is not enough — the plugin and its `/turbofit` command are not registered
until the process restarts.

## 4. Run the commands

1. `/turbofit status` — reads the **real** provider / gateway / runtime state for this machine
   (live config and routes, not canned data).
2. `/turbofit setup` — scans the machine, populates the model list for your hardware tier, and opens
   Hermes Desktop → Turbofit.

The `s0uthpaw` pet is selected for you (or run `hermes pets select s0uthpaw` manually).

## Still says `not a … command: turbofit`?

That profile has not reloaded plugins. Install and enable it **in that profile**, then restart:

```bash
hermes -p <profile> plugins install --enable https://github.com/SouthpawIN/turbofit.git
```

Then fully quit (⌘Q) and reopen that profile. Desktop profile sessions only scan that profile's
`plugins/` directory and `plugins.enabled`, so each isolated profile (TurboSouth included) needs its
own install.
