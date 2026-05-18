# interface-design system

> Tracked patterns for the Commit Type Classifier. Source of truth lives
> at `docs/design-system.md` — this file is a quick-glance index for the
> skill. Update both when patterns evolve.

## Direction & feel

- **Identity:** dev-tool dashboard wearing IDE / terminal as its visual language.
- **Audience:** software engineers + academic jury.
- **Feel:** terminal-precise, mono-leaning, hierarchy through alignment + a single accent per row.

## Strategy decisions

| Decision | Choice |
|---|---|
| Depth | Borders-only. No drop shadows (lose contrast in dark mode). |
| Spacing base | 4 px. Scale: 4 · 8 · 12 · 16 · 24 · 32. |
| Radius | 4 (inputs/chips) · 8 (cards) · 12 (modals). Sharper = technical. |
| Surface temperature | Slate. Only lightness shifts (`#0F172A → #1E293B → #334155`). |
| Type pair | **Fira Sans** (UI) + **Fira Code** (data + class labels). |
| Motion | 120ms (`m-fast`) and 200ms (`m-base`), deceleration easing. Color/opacity only — never scale or translate. |

## Class-color palette (signature)

Each Conventional Commit class is an IDE syntax token:

```
feat      #22C55E   added-line green
fix       #EF4444   removed-line red
docs      #38BDF8   doc-link sky
refactor  #A78BFA   keyword violet
test      #FBBF24   warning amber
```

The five class colors are the **only** accents in the entire app. Plus
the `ring-focus` sky (`#38BDF8`) for keyboard focus and active-tab.

## Signature element

**2px left gutter on every prediction card**, colored by predicted
class. Mirrors the IDE margin marker next to modified lines on a code
review. The class label below renders as a monospace chip in the same
color. Recurring components: predict card, repository row, history row.

## Core patterns (location in code)

| Pattern | CSS class | Use |
|---|---|---|
| Prediction card | `.predict-card` (`--class-color` set inline) | Predict tab — main result. |
| IDE-token chip | `.chip-{feat\|fix\|docs\|refactor\|test}` | Any class label inline. |
| Probability / histogram bar | `.bar-row` + `.bar-track` + `.bar-fill` | Predict probabilities, Repo distribution, Metrics F1. |
| Diff viewer | `.code` + `.l-add` / `.l-del` / `.l-ctx` | Predict tab when diff is present. |
| Metric card | `.metric-card` (`.is-leader` for top model) | Metrics tab — 5-column grid. |
| Tab strip | `.stTabs` overridden | All tabs. |
| Eyebrow label | `.eyebrow` | Section markers, UPPERCASE 0.06em. |

## Anti-patterns we explicitly reject

- Drop shadows or gradients.
- Pie charts (we use horizontal bars sorted desc).
- Emoji as iconography.
- Multiple accent hues (only the 5 class colors + focus sky).
- Different hues per surface (same slate, only lightness changes).
- Scale/translate on hover.
- Rounded "pill" tabs.

## Streamlit-specific bindings

- `.streamlit/config.toml` provides the dark base + primary `#38BDF8`.
- `app/streamlit_app.py` injects all `:root` variables and component
  styles via a top-level `st.markdown(THEME_CSS, unsafe_allow_html=True)`.
- All custom components are HTML strings built by helpers
  (`predict_card`, `bar_row`, `histogram_row`, `chip`, `metric_card`,
  `render_diff_block`).

## Token files

- Full token spec → `docs/design-system.md`
- Streamlit theme   → `.streamlit/config.toml`
- Injected CSS     → `THEME_CSS` constant in `app/streamlit_app.py`

When tokens change: update all three.
