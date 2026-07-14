# Night Research Institute Full UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the codex-with-cc implementer/reviewer/final-verifier chain. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate the five approved Image2 screens into the existing production frontend while preserving all current RAG, session, attachment, document, pipeline, and source-review behavior.

**Architecture:** Keep the established single-file, dependency-free frontend in `src/app/static/index.html`. Treat the five reference images plus `docs/superpowers/specs/2026-07-14-night-research-image-analysis.md` as the visual source of truth; implement one shared application shell and stateful view layouts rather than five duplicate pages. Extend `tests/test_static_frontend.py` with stable structural and safety contracts before changing production markup/CSS/JavaScript.

**Tech Stack:** HTML5, CSS custom properties/grid/flex/keyframes, inline SVG, vanilla JavaScript DOM APIs, FastAPI static serving, pytest/BeautifulSoup, Node syntax validation, Chrome/Playwright visual smoke checks.

---

### Task 1: Lock the Five-Screen Contract

**Files:**
- Modify: `tests/test_static_frontend.py`
- Read: `src/app/static/index.html`

- [ ] Add tests requiring the nocturne token names, chat-first default route, semantic brand/header elements, evidence constellation, Research Index, project overview, library archive layout, pipeline overview/events, source workspace, reduced motion, responsive breakpoints, and safe citation DOM rendering.
- [ ] Run `python -m pytest tests/test_static_frontend.py -q` and record the expected failures from the new contract.
- [ ] Keep every pre-existing assertion for API endpoints, ids, readable Chinese text, session/document scope, attachment behavior, deletion, and source handling.

### Task 2: Implement the Shared Shell and Chat States

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] Replace the old palette with the semantic colors in the image-analysis spec while maintaining compatibility aliases for existing selectors.
- [ ] Rebuild the sidebar and topbar to match the common 277px/70px visual skeleton, including an inline SVG lighthouse mark, view label/line, service state, navigation, history and fixed project switcher.
- [ ] Make chat the implicit default route while preserving explicit `#files` and `#runs`.
- [ ] Build the open editorial welcome composition from Image 1: left title/copy/index, sparse SVG/CSS evidence constellation, right project overview, bottom composer and example prompts.
- [ ] Restyle live/historical conversation output to Image 2: open user question, collapsible compact trace summary, wide article answer, safe inline citation markers, numbered sources and faded context visualization.
- [ ] Preserve attachment upload/delete, mode choice, SSE steps, new chat, session restoration, document scope, Enter/Shift+Enter and IME behavior.

### Task 3: Implement the Document Archive

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] Convert the document page from intro cards to the Image 3 open archive structure.
- [ ] Keep the upload form ids, project/new-topic fields and upload endpoint; present them in the left operation rail.
- [ ] Render recent documents as index rows with sequence, title/summary, filename, time, status, evidence count when available, document chat and delete.
- [ ] Keep all existing project refresh and upload completion behavior.

### Task 4: Implement the Run Dashboard and Source Workspace

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] Recompose run metrics and the run list to Image 4 with open divisions, tabular numbers and a right project overview/recent-events rail.
- [ ] Preserve topic/project data and topic-chat entry; surface it in the overview or a compact secondary region instead of deleting functionality.
- [ ] Convert `#sourceDrawer` into the Image 5 integrated split workspace: source metadata, PDF, new-window action and selectable chunk list.
- [ ] Ensure opening source compresses the run area and closing source restores the dashboard.

### Task 5: Responsive, Accessibility and Motion Pass

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] Add 1200px, 900px, 768px and 480px behavior derived in the image-analysis spec.
- [ ] Add visible focus states, adequate contrast, 44px mobile touch targets and semantic labels.
- [ ] Keep motion event-driven; use only transform/opacity for the constellation and view entrances.
- [ ] Disable decorative animation under `prefers-reduced-motion: reduce`.

### Task 6: Verify and Compare

**Files:**
- Inspect: `src/app/static/index.html`, `tests/test_static_frontend.py`
- Reference: `tmp/ui-references/night-research/*.png`

- [ ] Run `python -m pytest tests/test_static_frontend.py -q`.
- [ ] Run `python -m pytest tests/test_static_frontend.py tests/test_app_startup.py -q`.
- [ ] Run `python -m pytest tests/test_static_frontend.py::test_inline_script_is_valid_javascript -q`.
- [ ] Run `python -m pytest -q`.
- [ ] Run `git diff --check` and review changed files for scope.
- [ ] Capture and visually inspect chat welcome, chat answer, document library, run dashboard and source workspace at 1680×945; also inspect 1024×768 and 390×844 for overflow.

