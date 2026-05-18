# Design System — Commit Type Classifier

> **Direction.** A dev-tool dashboard that wears the IDE / terminal as its
> visual language. Cards feel like inline code review panes; class labels
> behave like syntax tokens; probabilities read like a build profiler.
> The interface is invisible — what stays in your eye is the data.

---

## 1. Intent

| Question | Answer |
|---|---|
| **Who** | A software engineer (or jury) inspecting whether an ML model classified a commit correctly. They live in IDEs and `git log`. |
| **What must they do** | Read one prediction at a glance, scan a 100-commit repo audit, compare 5 models. |
| **Feel** | Terminal-precise. Mono-leaning. No decoration. Hierarchy through monospace alignment and a single accent per row, not boxes-within-boxes. |

---

## 2. Tokens

### 2.1 Surfaces (whisper-quiet elevation; ~5% lightness per step)

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#0F172A` | Page canvas (slate-900). |
| `--surface-1` | `#1E293B` | Default card surface. |
| `--surface-2` | `#334155` | Elevated: dropdowns, modals, hover. |
| `--surface-inset` | `#0B1220` | Inputs, code blocks — slightly DARKER than `--bg` so they read as "type here". |

### 2.2 Borders (low-opacity, structural)

| Token | Hex / rgba | Role |
|---|---|---|
| `--border-soft` | `rgba(148,163,184,0.10)` | Default card separation. |
| `--border` | `rgba(148,163,184,0.18)` | Standard divider. |
| `--border-strong` | `rgba(148,163,184,0.28)` | Emphasis (active tab, focus). |
| `--ring-focus` | `#38BDF8` (40% alpha overlay) | Keyboard focus only. |

### 2.3 Text hierarchy (four tiers, not two)

| Token | Hex | Role |
|---|---|---|
| `--ink` | `#F8FAFC` | Headlines, predicted class label. |
| `--ink-2` | `#CBD5E1` | Body and primary data. |
| `--ink-3` | `#94A3B8` | Metadata (timestamps, model name). |
| `--ink-mute` | `#64748B` | Placeholder, disabled. |

### 2.4 Class colors — the signature

Each Conventional Commit class maps to an IDE syntax token. **These five
colors are the ONLY accents in the app.** No others.

| Class | Hex | IDE analogy |
|---|---|---|
| `feat` | `#22C55E` | "added" diff line |
| `fix` | `#EF4444` | error / "removed" diff line |
| `docs` | `#38BDF8` | comment, doc-link (sky) |
| `refactor` | `#A78BFA` | keyword (violet) |
| `test` | `#FBBF24` | warning glyph (amber) |

Use the matching **8% alpha tint** (`#22C55E14`, etc.) for the soft
backgrounds of chips and probability-bar tracks.

### 2.5 Spacing scale (4-unit base, multiples only)

| Step | px | Use |
|---|---|---|
| `s-1` | 4 | Icon ⇄ label gap. |
| `s-2` | 8 | Inline elements inside a chip. |
| `s-3` | 12 | Field inner padding (top/bottom). |
| `s-4` | 16 | Card inner padding, paragraph rhythm. |
| `s-5` | 24 | Card-to-card gap. |
| `s-6` | 32 | Section-to-section break. |

No values outside this scale. Random padding signals "no system".

### 2.6 Corner radius

| Step | px | Use |
|---|---|---|
| `r-1` | 4 | Buttons, inputs, chips. |
| `r-2` | 8 | Cards, code blocks. |
| `r-3` | 12 | Modals, full-page panels. |

Sharper than typical: technical, not friendly.

### 2.7 Elevation (borders-only strategy — dark mode rule)

We **do not** use drop shadows. In dark mode, shadows lose contrast and
look smudgy. Hierarchy emerges from surface lightness + border.

| Level | Treatment |
|---|---|
| `e-0` | `--bg`, no border. |
| `e-1` | `--surface-1` on `--bg`, `--border-soft` 1px. |
| `e-2` | `--surface-1` on `--bg`, `--border` 1px + left **2px class-color gutter** (the signature). |
| `e-3` | `--surface-2` on `--bg`, `--border-strong` 1px (hover/focus, dropdowns). |

### 2.8 Typography

| Family | Role |
|---|---|
| **Fira Sans** | Display + body (UI labels, paragraphs). |
| **Fira Code** | All data: predicted class chip, probabilities, hashes, file paths, diff text, table cells with numbers. |

Type scale (rem; 16px root):

| Token | rem | Weight | Letter-spacing | Use |
|---|---|---|---|---|
| `t-display` | 1.75 | 600 | -0.01em | Tab titles. |
| `t-h2` | 1.25 | 600 | -0.005em | Card titles. |
| `t-body` | 0.9375 | 400 | 0 | Body paragraphs. |
| `t-label` | 0.8125 | 500 | 0.02em UPPER | Section eyebrows, table headers. |
| `t-data` | 0.875 | 500 | 0 | Mono numerics (use `font-variant-numeric: tabular-nums`). |
| `t-meta` | 0.75 | 400 | 0.01em | Timestamps, hashes. |

Line height 1.5 for body, 1.3 for headings, 1 for mono data rows.

### 2.9 Motion

| Token | ms | Easing |
|---|---|---|
| `m-fast` | 120 | `cubic-bezier(0.2, 0, 0, 1)` (decel) |
| `m-base` | 200 | same |

No spring, no bounce. Hover changes color/opacity, **never** scale or translate.

---

## 3. Component patterns

### 3.1 Prediction card (e-2 + class gutter — the signature)

```
┌─ 2px gutter in class color ──┐  ← left edge only
│ feat                  92.4%  │  ← Fira Code label, tabular-nums %
│ ────────────────────────────│
│ baseline_tfidf · 124 ms      │  ← model · latency, t-meta
│                              │
│ ▁▁▁ feat     ▓▓▓▓▓▓▓▓▓▓ 0.924│  ← probability bars (see 3.4)
│ ▁▁▁ fix      ▓▓        0.040 │
│ ▁▁▁ docs     ▓         0.018 │
│ ▁▁▁ refactor ▓         0.012 │
│ ▁▁▁ test     ▓         0.006 │
└──────────────────────────────┘
```

CSS recipe:
```css
.predict-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-left: 2px solid var(--class-color); /* feat=green, fix=red, ... */
  border-radius: var(--r-2);
  padding: var(--s-5);
}
.predict-card .label  { font-family: 'Fira Code'; color: var(--class-color); }
.predict-card .conf   { font-variant-numeric: tabular-nums; color: var(--ink); }
```

### 3.2 Class chip — like a syntax token in `git log`

```
[ feat ]    [ fix ]    [ docs ]
```

```css
.chip {
  font-family: 'Fira Code';
  font-size: var(--t-meta);
  padding: 2px 8px;
  border-radius: var(--r-1);
  background: color-mix(in srgb, var(--class-color) 12%, transparent);
  color: var(--class-color);
  border: 1px solid color-mix(in srgb, var(--class-color) 30%, transparent);
}
```

### 3.3 Code block (diff or message)

Surface is `--surface-inset` (darker than canvas, the "type here" cue).
Diff lines preserve their `+`/`-` prefix; we color the entire line by
prefix using `feat` green and `fix` red. **No syntax highlighting beyond
diff prefix** — keep it readable.

```css
.code {
  background: var(--surface-inset);
  border: 1px solid var(--border-soft);
  border-radius: var(--r-2);
  padding: var(--s-4);
  font-family: 'Fira Code';
  font-size: 0.8125rem;
  line-height: 1.55;
  color: var(--ink-2);
  white-space: pre;
  overflow-x: auto;
}
.code .l-add { color: #4ADE80; }     /* slightly lighter than feat for readability */
.code .l-del { color: #FCA5A5; }     /* slightly lighter than fix for readability */
```

### 3.4 Probability bar (the chart signature)

Horizontal track. Right-aligned `tabular-nums` value. No gradient, no
animation longer than `m-base`. Bars render in their **own class color**;
this is the moment the class colors carry meaning.

```
feat      ████████████████████  0.924
fix       █                      0.040
docs      ▌                      0.018
refactor  ▌                      0.012
test      ▌                      0.006
```

```css
.bar-row {
  display: grid;
  grid-template-columns: 5rem 1fr 4ch;
  align-items: center;
  gap: var(--s-3);
  font-family: 'Fira Code';
  font-size: var(--t-data);
}
.bar-track {
  height: 8px;
  background: color-mix(in srgb, var(--class-color) 10%, transparent);
  border-radius: 2px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: var(--class-color);
  transition: width var(--m-base) cubic-bezier(0.2,0,0,1);
}
```

### 3.5 Metric card (Metrics tab — 5-model grid)

Compact. Each card pinned to one model. Border-left **2px** in a
neutral slate (`#334155`) — NOT class color, because metrics belong to
the model, not the class. Big metric in `t-display`, mono, tabular.

```
┌─ 2px slate gutter ─────────┐
│ baseline_tfidf      71.0%  │  ← acc as big number, mono tabular
│ ──────────────────────────│
│ macro F1     0.6632        │
│ weighted F1  0.7187        │
│ precision    0.6234        │
│ recall       0.7221        │
└────────────────────────────┘
```

### 3.6 Repo audit histogram (Repository tab)

Same `.bar-row` pattern as 3.4, but each bar is a count instead of a
probability. Sort **descending by count**. Right-aligned counts in
mono. The model's confidence on each bucket appears as a thinner second
bar below the count in the same color at 30% alpha — invisible at a
glance, present when you look for it.

### 3.7 Tables (comparison + history)

Borderless. Row separator = `--border-soft` 1px on bottom. Header row:
`t-label` UPPER. All numeric columns mono+tabular. Hover row:
`--surface-2`. No zebra striping.

### 3.8 Tab strip

Flat. Active tab: bottom border 2px in `--ring-focus` color (`#38BDF8`).
No fills, no rounded "pill" tabs.

```css
.tab          { padding: var(--s-3) var(--s-4); color: var(--ink-3); border-bottom: 2px solid transparent; }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: #38BDF8; }
```

---

## 4. States (mandatory — missing states feel broken)

| State | Treatment |
|---|---|
| Hover (interactive) | Background → `--surface-2`. Never scale/translate. |
| Active/Pressed | Background → `--surface-2`, border → `--border-strong`. |
| Focus (keyboard) | 2px outline using `--ring-focus`, 2px offset. |
| Disabled | Color → `--ink-mute`, opacity 0.6, cursor `not-allowed`. |
| Loading | Skeleton blocks in `--surface-2` with shimmer at `m-base`. |
| Empty | Single line in `--ink-3`: `"no commits scanned yet — try /Users/.../parqueaderos-api"`. |
| Error | Inline strip with `fix`-red 12% alpha background + `fix`-red text. |

---

## 5. Streamlit translation map

Streamlit's theme accepts `[theme]` in `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#38BDF8"          # focus ring / accent / active tab
backgroundColor = "#0F172A"       # --bg
secondaryBackgroundColor = "#1E293B"  # --surface-1
textColor = "#F8FAFC"             # --ink
font = "sans serif"               # we override to Fira Sans via CSS
```

Then inject `<style>` in `streamlit_app.py` to:
1. Import Fira Sans + Fira Code from Google Fonts.
2. Bind the CSS variables in this doc onto `:root`.
3. Override `.stApp`, `.stTabs`, `.stMetric`, `.stDataFrame` to use them.
4. Add the `.predict-card`, `.chip`, `.bar-row`, `.code` classes for
   custom `st.markdown(unsafe_allow_html=True)` blocks.

---

## 6. The signature, restated

If you stripped every other choice and kept only **one**, the
identifying mark of this app would be:

> The **2px left gutter** on the prediction card, in the predicted
> class's syntax-token color, with the class label rendered as a
> monospace chip directly below it.

Anywhere a prediction appears — Predict tab, Repository scan rows,
History list — that gutter shows up. It is the IDE-margin-marker of
this product.

---

## 7. Do-not list

- No emoji icons. SVG only (Lucide or Heroicons), single-set.
- No drop shadows in dark mode — use border + surface lift.
- No multiple accent colors. Only the five class colors plus the focus sky.
- No different hues per surface — same slate, only lightness changes.
- No scale/translate on hover. Color/opacity only.
- No rounded "pill" tabs. Flat with bottom-border activation.
- No pie charts. Horizontal bars sorted descending only.
