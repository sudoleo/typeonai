# consens.io LinkedIn content guide

Persistent editorial and visual rules for weekly consens.io posts. These rules
incorporate the owner's feedback and apply to every future draft.

## Language and positioning

- Publish in polished, native English unless explicitly requested otherwise.
- Build each post around trust, meaningful model differences, and decisions
  where accepting the first confident answer could be risky or costly.
- Use one strong idea per post. Do not turn a post into a feature list.
- Prefer examples with real consequences. Avoid trivia unless the disagreement
  reveals an important measurement, evidence, or decision problem.
- The recurring core is: confidence is not consensus. Do not repeat this exact
  sentence mechanically; express the idea through evidence.
- Write like a thoughtful founder or product researcher, not a generic AI
  marketing account. Avoid inflated claims, canned hooks, and AI clichés.

## Brand accuracy

- Use only the real consens.io mark and wordmark from the current project.
  Never invent a logo, icon, symbol, wordmark, or brand color.
- Read the current public templates and CSS before designing. Treat
  `static/css/public-tokens.css`, `static/css/landing.css`, and
  `static/css/components-consensus-insights.css` as the visual source of truth.
- Product mockups must match the current product, including copy, scores,
  provider assignments, marker color, line style, thickness, and offset.
- Marker styling is not remembered from an older post. It is inspected in the
  current CSS every time. As of 2026-07-29, markers use solid underlines:
  1 px for a minor or split signal and 2 px amber for a major contradiction.

## Art direction

- Prefer real product evidence, authentic screenshots, and editorial crops over
  decorative illustrations or invented dashboards.
- Aim for restrained editorial design: clear hierarchy, generous whitespace,
  thin rules, purposeful asymmetry, and limited accent color.
- Avoid the common AI-generated SaaS look:
  - oversized generic slogans dominating the image;
  - grids of symmetrical rounded cards;
  - gradients, glows, excessive shadows, and decorative badges;
  - provider bubbles used as decoration;
  - too many labels explaining what the viewer can already see;
  - fake product chrome or UI that does not exist.
- If a custom composition is necessary, it should feel like a product editorial
  or research note, not a template-generated infographic.
- Use the 4:5 LinkedIn format at 1080 × 1350 unless the chosen format requires
  something else.

## Content formats

- Rotate between real product video, editorial product crop, flowchart,
  What’s New, model disagreement, benchmark evidence, and founder learning.
- Do not force a format. Use the format that best proves the week's single idea.
- A video should show the actual product flow, use concise captions, and avoid
  synthetic presenter footage.

## Product video

The demo video is generated, not hand-edited. Pipeline and flags:
`recording/README.md` (local tooling, gitignored). Rules that apply to every
cut:

- Record the running app, never a rebuild of it. The recorder drives the real
  interface through the app's own demo scenario (`static/demo.js`); the shell,
  the run, the consensus, the claim badges and the difference cards are the
  product's own DOM.
- Every scene reacts to real DOM state instead of a guessed duration, so a
  product change shows up in the next render rather than drifting out of sync.
- Default output: 1080 × 1350 (4:5), ~50–55 s, 30 fps, dark theme. A 1920 × 1080
  landscape master is available for the site and YouTube.
- One idea per cut, six captions at most. A caption names what the product is
  doing in the moment it does it; it never claims something the frame does not
  show.
- Framing: the product's reading column sets the zoom, then the camera settles.
  Never crop words. Move attention with a spotlight (dim everything but one
  element) and with copy, not with constant zooming.
- The stage around the product is monochrome and built from the same five
  surface values as the app. Colour belongs to the agreement signals only.
- Disclose the setup on the end card (`Demo scenario · real interface`). The
  interface is real; the model answers in the demo scenario are canned, and
  that difference is never blurred.
- Silent by default. Feed video is watched muted; captions carry it.

## Fact and visual QA

- Verify every number, model assignment, quote, capability, and result against
  the current website, code, benchmark data, or a documented real run.
- For medical, legal, financial, or similarly sensitive examples, frame model
  outputs as evidence of disagreement, not advice, and include the appropriate
  verification note.
- Render every final visual and inspect it at full size before delivery.
- Check the real logo, spelling, line style, sharpness, spacing, text wrapping,
  contrast, clipping, and mobile-feed readability.
- Nothing is published without explicit owner approval.

## Learning loop

- Treat every owner comment as a standing rule unless it is explicitly limited
  to one post.
- Record material new feedback in this guide before preparing the next post.
- Ask for impressions, clicks, profile visits, reactions, comments, and saves
  after publishing. Use those results to adjust future hooks, formats, and CTAs.
