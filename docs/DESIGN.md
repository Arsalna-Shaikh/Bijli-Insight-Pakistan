# Design: The Monthly Bill

The dashboard is issued as a Pakistani electricity bill (K-Electric / LESCO / IESCO), the one energy document every household reads. Source of truth: `index.html`.

## Principles
- A printed form, not an app: ruled ink lines, no cards, no shadows, no border radius.
- Every headline figure sits next to its evidence (q10–q90 range, baseline, source file).
- Real data only. Loading, empty and error states say what is happening; never show placeholder numbers.

## Tokens (`:root`, with a dark "night copy" set)
| Role | Light | Dark |
|---|---|---|
| Paper | `--paper #FCFCFA` | `#14161B` |
| Ink / secondary | `--ink #17181A` / `--ink2 #4A4D52` | `#ECEDEF` / `#A7ABB3` |
| Hairline / hover | `--hair #C9CBCF` / `--soft #EEEFEC` | `#363940` / `#1C1F25` |
| Bill green tint | `--green #DCEBD9`, text `--green-ink #24563A` | `#17271E` / `#9FD3B0` |
| Bill pink tint | `--pink #F6DDE3`, text `--pink-ink #8A2340` | `#2A1820` / `#F2A9BC` |
| Stamp (actions, links, VERIFIED) | `--stamp #233E8B` | `#9DB2F2` |
| Heat / alert | `--alert #B3261E` | `#FF8A80` |
| Cool end of heat ramp | `--cool #2F5FA8` | `#8FB3F0` |

Tinted fields (`.tint`, `.tint.p`) carry the security-guilloche pattern from `--guil`.

## Type
- Archivo (variable width): body at wdth 100; labels and headings uppercase at wdth 70–80, tracked +.04–.06em.
- Martian Mono: every figure (temperatures, rupees, times, hashes), tabular.
- Noto Nastaliq Urdu: bilingual section labels and the wordmark بجلی بصیرت پاکستان.
- Three working sizes (12–13 / 15 / display). Rank comes from weight, case and rule, not from more sizes.

## Rules and structure
- Outer frame and section dividers 1.5px ink; inner dividers 1px ink; line items 1px `--hair`; totals use a 1.5px rule above and a 3px double rule below.
- Layout: masthead (brand | consumer panel | pipeline status + stamp), bilingual 5-tab strip, ruled blocks (`.blk`) in one- or two-column `.split`s, and a dashed tear-off provenance stub as the footer.

## Signature components
- `.pay`: the "amount payable" box. Pink tint, display figure, range line. Used for tomorrow's high and the bill total.
- `.items`: bill line items with right-aligned mono amounts and a double-ruled total.
- `.hist`: the consumption-history strip (7-day low–high bars on a tick scale, red when ≥ 35 °C).
- `.stampmark`: one rotated rubber stamp per screen (pipeline status).
- Controls: square-cornered ink buttons (`.btn`, `.btn.solid`), segmented `.seg`, square check boxes in `.opts`.

## Motion
One authored moment only: the payable figure "prints" left to right (`clip-path`, 14 steps) when the forecast loads. It is visible by default and disabled under reduced motion.
