# Website design tokens

The project site (`docs/`) uses tokens adapted from a fintech design system in the
[DesignMD](https://designmd.me/discover) catalogue (retrieved with the DesignMD connector, November 2026).
Only colours, type scale, radii and shadows were borrowed; no third-party branding is used.

| Role | Light | Dark |
|---|---|---|
| Ink (headings) | `#0a2540` | `#f0f4fa` |
| Body text | `#425466` | `#c3cbd9` |
| Muted text | `#64748d` | `#8b97ab` |
| Accent (links, active controls) | `#533afd` | `#8f80ff` |
| Page / alt section / card | `#ffffff` / `#f6f9fc` / `#ffffff` | `#0a1020` / `#0d1528` / `#111b31` |
| Border | `#e5edf5` | `#1f2b45` |

- **Type:** Inter 300–700 (body 16px / 1.65), JetBrains Mono for numbers. The source system's proprietary face is replaced by Inter.
- **Radii:** 4px controls, 6–8px cards, pill buttons. **Shadows:** two-layer (`rgba(50,50,93,.18) 0 13px 27px -5px, rgba(0,0,0,.16) 0 8px 16px -8px`).
- **Chart series** (one colour per model, everywhere): actual `#0a2540`, A `#8a8986`, B `#2a78d6`, C `#eb6834`,
  lender `#1baf7a`, D `#eda100`, D-base `#6250d6`. These slots come from a colour-blind-validated categorical palette.
  Every chart with two or more series has a legend, and none uses a second y-axis.
