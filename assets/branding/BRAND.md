# C5 · Crunch AI / Data Science — Brand Guide

> **Voice:** rigorous, sober, allergic to hype. The kind of voice a senior data scientist trusts.
> **Feel:** a working notebook. Quiet, ordered, replicable.

Extends the family brand (see `../../assets/brand/BRAND-FAMILY.md`). This file only documents the C5-specific overrides.

---

## Identity

- **Full name:** Crunch AI / Data Science
- **Program code:** C5
- **Full title in copy:** *C5 · Crunch AI / Data Science*
- **Tagline (short):** Data-science honesty, in twelve weeks.
- **Tagline (long):** A free, open-source twelve-week data-science and classical-ML track for engineers who already know Python — NumPy through PyTorch, with the experimentation discipline employers actually want.
- **Canonical URL:** `codecrunchglobal.vercel.app/course-c5-aidatascience`
- **License:** GPL-3.0

---

## Where C5 diverges from the family palette

C5 inherits Ink/Parchment/Gold and adds a single restrained accent for charts, model-output highlighting, and the "experiment-tracker" UI feel:

| Role | Name | Hex | Use |
|------|------|-----|-----|
| Accent | Plot Violet | `#7C3AED` | Chart highlights, learning-rate curves, the C5 mark |
| Accent deep | Plot Violet deep | `#5B21B6` | Hover states, callouts on parchment |
| Accent soft | Plot Violet soft | `#DDD6FE` | Subtle row highlights in DataFrame screenshots |

```css
:root {
  --plot:        #7C3AED;
  --plot-deep:   #5B21B6;
  --plot-soft:   #DDD6FE;
}
```

> **Why violet, not the usual data-viz cyan?** Every commodity ML tool uses cyan or the matplotlib default blue. The brand is supposed to read as *editorial* — closer to the violet of an old academic publisher than the cyan of a SaaS dashboard.

### Typography

Same as family. EB Garamond display, Lora body, **JetBrains Mono for any rendered DataFrame, code, or numeric column** (this is the C5 mono convention — column-aligned numbers must use mono).

---

## The "experiment card" pattern

C5 introduces one recurring page element: the **experiment card**. Every model the curriculum builds has one:

```
┌─────────────────────────────────────────────────┐
│  EXPERIMENT  W04-01-housing-linear              │
│                                                 │
│  Goal:     Predict house prices                 │
│  Data:     Ames housing, n=1460                 │
│  Method:   Linear regression, L2 regularized    │
│  CV:       5-fold                               │
│  Metric:   RMSE                                 │
│  Result:   $28,450 (baseline: $42,180)          │
│  Status:   ✓ better than baseline                │
└─────────────────────────────────────────────────┘
```

Rules:

- Always JetBrains Mono.
- Always exactly these eight rows (goal, data, method, CV, metric, result, status; "notes" optional 9th).
- Status is one of: `✓ better than baseline`, `✗ worse than baseline`, `≈ ties baseline`, `🚧 in progress`.
- Cards are committable: they're 10-line markdown blocks in the experiment log.

This pattern is what makes the curriculum feel *like a working data team*, not a tutorial.

---

## Voice rules (extending family)

C5 holds the family voice and adds:

- **Numbers cited, not asserted.** "The model is more accurate" is wrong; "the model's RMSE drops from 42k to 28k" is right.
- **Confidence proportional to evidence.** "We see 60% accuracy on a held-out test set, which beats the 51% baseline but is not state of the art." Not "our model achieves great accuracy."
- **No "ML magic" language.** No "the AI learns," no "the model figures out." Models *fit*, *predict*, *optimize a loss*. Be specific.
- **Acknowledge bias and limitations every time we ship a model.** Always a final paragraph: "What this model doesn't know, who could be harmed by its errors, and what we'd validate before deploying it."

---

## Course page conventions

For `course-c5-aidatascience.html`:

- Editorial parchment hero with a single violet curve graphic (a synthetic learning-rate curve).
- The 4-phase ladder rendered as a "training-progress" bar — each phase a discrete chunk like an epoch.
- Each capstone deliverable shown as an experiment card.
- A small "honesty section" near the footer addressing what C5 does NOT teach (LLMs, RL, distributed training) with links to where each is covered.

---

*GPL-3.0. Fork freely.*
