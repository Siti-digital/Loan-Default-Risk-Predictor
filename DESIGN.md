# Website design tokens

The project site (`docs/`) follows the visual style of a reference page chosen by the author
(the recovery calculator at charterflightsupport.com): warm cream canvas, deep navy panels,
fuchsia highlights, a lime call-to-action, square corners with small fuchsia corner markers,
and a serif display face over a clean sans. Only colours, type pairing and component shapes
were borrowed; no logos, imagery or text from that site are used.

| Role | Value |
|---|---|
| Canvas / alternate section | `#f1eee9` / `#f7f5f1` |
| Card surface | `#fcfcfc`, 1px border `#e2e0dc`, radius 0 |
| Navy (headings, panels) | `#0b1b4d` (inner tiles `#172a63`) |
| Fuchsia (accent, highlight tiles, markers) | `#ed008c` (text on cream `#c40074`) |
| Lime (call-to-action, positive tags) | `#cff564` |
| Body text / muted | `#252d4f` / `#61688a` |

- **Type:** Fraunces 600–700 for headings, big numbers and labels; Albert Sans 400–700 for body, controls and tables (tabular numbers); JetBrains Mono only for code.
- **Components:** square segmented controls (active = navy fill), bordered chips (selected = navy outline + grey fill), grey-filled inputs, navy result panels with a fuchsia feature tile, 6px fuchsia squares on panel corners and as bullets.
- **Chart series** (one colour per model in the site and the notebook figures): actual `#0b1b4d`, A `#8a8986`, B `#2a78d6`, C `#d4007d`, D `#eda100`, lender `#1baf7a`, D-base `#6250d6`.
  B, C, D and lender together pass the colour-vision-deficiency check (all pairs, worst ΔE 9.1 on `#fcfcfc`).
  Every multi-series chart has a legend, and no chart uses a second y-axis. The site has a single light theme, like the reference.
