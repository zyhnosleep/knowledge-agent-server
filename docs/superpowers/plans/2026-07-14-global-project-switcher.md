# Global Project Switcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent global project switcher matching the approved Night Research dashboard image.

**Architecture:** Keep project state in the existing `activeProjectSlug`/`activeProjectName` variables and make `setActiveProject` the only user-triggered switch path. Render two synchronized controls from the existing `/api/projects` response and persist only the selected slug in `localStorage`; page-specific loaders continue to build scoped API URLs from the active slug.

**Tech Stack:** Static HTML/CSS/JavaScript, FastAPI static shell, pytest/BeautifulSoup, browser smoke checks.

---

### Task 1: Lock the DOM and state contract

**Files:**
- Modify: `tests/test_static_frontend.py`
- Test: `tests/test_static_frontend.py`

- [ ] Add failing tests requiring `#sidebarProjectSelect`, `#projectSwitchTabs`, `#projectSyncStatus`, `aria-selected`, the storage key `night-research-active-project`, and a single `setActiveProject` switch path.
- [ ] Run `D:/Miniconda3/python.exe -m pytest tests/test_static_frontend.py -q` and confirm failures identify the missing switcher contract.

### Task 2: Add the approved project controls

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] Replace the sidebar project label with a native select inside `#sidebarProjectSwitcher` while retaining `#activeProjectLabel` as the visible selected-project value.
- [ ] Add `#projectSwitchBar` between the run-page introduction and metrics, with `#projectSwitchTabs` and `#projectSyncStatus`.
- [ ] Add open, border-led styles matching the approved image: 48px tab rail, signal underline, 10px monospace status, no large project cards.
- [ ] Add responsive behavior so tabs scroll at 768px and the sidebar control disappears with the sidebar.

### Task 3: Synchronize and persist project state

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] Implement `getStoredProjectSlug`, `storeActiveProjectSlug`, `renderProjectSwitchers`, and `setProjectSyncState` using safe DOM APIs.
- [ ] Update `loadProjects` to restore a valid stored slug before rendering page data.
- [ ] Update `setActiveProject` to persist, render both controls, clear document/session/attachment scope, reset the source drawer, refresh the active page, and surface sync success/failure.
- [ ] Route sidebar changes and tab clicks through `setActiveProject` and keep keyboard/ARIA state synchronized.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Verify behavior and visual fidelity

**Files:**
- Modify only if verification exposes a defect: `src/app/static/index.html`, `tests/test_static_frontend.py`

- [ ] Run `D:/Miniconda3/python.exe -m pytest tests/test_static_frontend.py -q`.
- [ ] Run `D:/Miniconda3/python.exe -m pytest -q`.
- [ ] In a 1680×945 browser, select `agent` from both controls and verify metrics become 2/2/0 and the right rail reads `agent`.
- [ ] Reload and verify `agent` remains selected.
- [ ] At 390×844, verify chat, files, and runs have no horizontal page overflow.
- [ ] Commit, push `main`, fast-forward the server checkout, restart API, and confirm the forwarded `8011` page contains the switcher and reports health `ok`.
