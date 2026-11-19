# Support links — setup and wiring

Where the support links live, what each platform needs, and how they surface.
Nothing here is published until the account exists and the handle is confirmed.

## The four surfaces

| Surface | What it shows | Source of truth |
|---|---|---|
| GitHub **Sponsor** button | Patreon / Ko-fi / Buy Me a Coffee / GitHub Sponsors | `.github/FUNDING.yml` |
| README **Support** section | the same links, in prose, near the top | README (generated from the wiki hub) |
| **Linktree** | one page holding every link, for bios and video descriptions | linktr.ee profile |
| Plugin / dashboard footer | a quiet link, never a nag | dashboard bundle |

## What each platform needs

**Patreon** — recurring support, tiers. Create the page, pick tier names and
prices, then the handle is the slug in `patreon.com/<handle>`. Goes in
`FUNDING.yml` as `patreon: <handle>`.

**Ko-fi** — one-off tips, no account required for the supporter, 0% platform fee
on donations. Handle is `ko-fi.com/<handle>`. Goes in as `ko_fi: <handle>`.

**Buy Me a Coffee** — one-off and memberships, supporters need an account.
Handle is `buymeacoffee.com/<handle>`. Goes in as `buy_me_a_coffee: <handle>`.

**Linktree** — not a funding platform; it is the hub. GitHub has no native
Linktree key, so it goes in as a `custom:` URL. Put the other three on the
Linktree page too, plus the repos and the YouTube channel, so one URL works in a
bio.

**GitHub Sponsors** — optional, needs enrollment and a payout account. If
enabled, `github: [SouthpawIN]` adds it to the same button.

## Wiring, in order

1. Create the accounts, confirm each handle by opening its public URL.
2. Uncomment the matching lines in `.github/FUNDING.yml` — one platform per line.
3. Add a **Support** section to the README source
   (`wiki-topics/turbofit/README.md` on the bot-wiki `wiki` branch), then run
   `tools/sync-wiki-readme.sh`. Never hand-edit the generated README.
4. Build the Linktree page: the three tip jars, both repos, the channel.
5. Mirror the same block into bot-wiki so the fleet shares one support surface.

## Rules

- **No placeholder links in public files.** A commented line is honest; a link to
  an account that does not exist is not.
- **One handle, one spelling, everywhere.** Divergent handles across repos and
  bios are how supporters land on a stranger's page.
- **Support is an invitation, never a gate.** Nothing in TurboFit is paywalled,
  and no runtime path checks whether someone donated.
- Public attribution stays **sovthpaw**.
