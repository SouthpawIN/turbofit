# Turbofit 2.3 local H3 promo

Purpose-built commercial for the 2.3 chain. **New MiniMax H3 clips only.** Old 2.1/2.2 footage is not used.

Plot: one engineer, one lab, one motif. The chamber grows through hardware tiers, Ornith sheds experts under pressure, then five keyless Nous portals replace a slammed NIM gate.

| Clip | Style | Beat |
|---|---|---|
| `bonsai-8gb` | anime | 8GB Bonsai takes root |
| `unleashed-16gb` | cinematic CG | chamber expands to Unleashed IQ3 |
| `unleashed-24-95` | photoreal | 24–95GB Unleashed Q3_K_XL |
| `ornith-aux` | claymation | Ornith MoE offloads experts |
| `scale-down` | watercolor | experts → context → auto → Bonsai |
| `nous-keyless` | pixel art | NIM closes, five Nous portals open |

```bash
scripts/generate-h3-promo-clips --device cuda:1 --prompts promo/h3-prompts-2.3.json --output-dir promo/h3-local-2.3
scripts/build-promo-video-2.3
```
