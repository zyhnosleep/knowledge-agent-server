# Night Research Institute UI Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing RAG application open into a distinctive dark editorial conversation workspace with a restrained evidence constellation and safe inline citations, without regressing current behavior.

**Architecture:** Keep the dependency-free single-page architecture in `src/app/static/index.html`. Extend its semantic HTML, CSS tokens, native CSS/SVG motion, and plain JavaScript DOM rendering in place. Phase 1 changes the global shell and chat view while preserving all existing DOM ids and API calls used by the library and pipeline views.

**Tech Stack:** HTML5, CSS custom properties/keyframes, vanilla JavaScript DOM APIs, FastAPI static serving, BeautifulSoup/pytest static-contract tests, Node `--check`, Chrome/Playwright smoke screenshots.

---

### Task 1: Add the Phase 1 Static Contract

**Files:**
- Modify: `tests/test_static_frontend.py`
- Read: `src/app/static/index.html`

- [ ] **Step 1: Add failing theme, entry, constellation, motion, and citation tests**

Append these checks, reusing `_html()` and BeautifulSoup already imported:

```python
def test_nocturne_theme_and_conversation_first_entry() -> None:
    html = _html()
    assert "--night-950: #070a0e" in html
    assert "--signal: #b1ef63" in html
    assert 'var activeView = "chat"' in html
    assert 'setView((location.hash || "#chat").slice(1))' in html


def test_chat_welcome_has_editorial_constellation_structure() -> None:
    soup = BeautifulSoup(_html(), "html.parser")
    welcome = soup.select_one("#chatWelcome")
    assert welcome is not None
    assert welcome.select_one(".welcome-kicker") is not None
    assert welcome.select_one(".evidence-constellation") is not None
    assert len(welcome.select(".constellation-node")) >= 5
    assert "从一个问题开始研究" in welcome.get_text(" ", strip=True)


def test_nocturne_motion_has_reduced_motion_fallback() -> None:
    html = _html()
    assert "@media (prefers-reduced-motion: reduce)" in html
    assert ".evidence-constellation" in html
    assert "animation: none" in html


def test_citations_use_safe_dom_rendering() -> None:
    html = _html()
    assert "function renderCitedText" in html
    assert "document.createTextNode" in html
    assert "citation-marker" in html
    assert "data-citation-index" in html
    assert '$("answerText").textContent = stripCitationMarkers(text)' not in html
    assert 'body.textContent = stripCitationMarkers(content || "")' not in html


def test_nocturne_responsive_breakpoints_present() -> None:
    html = _html()
    assert "@media (max-width: 768px)" in html
    assert "@media (max-width: 480px)" in html
```

- [ ] **Step 2: Run tests and confirm the new contract fails**

Run: `python -m pytest tests/test_static_frontend.py -q`  
Expected: the new nocturne tests fail; existing static tests remain green.

- [ ] **Step 3: Confirm the existing inline script still parses**

Run: `python -m pytest tests/test_static_frontend.py::test_inline_script_is_valid_javascript -q`  
Expected: `1 passed`.

### Task 2: Implement the Nocturne Tokens, Shell, and Empty Chat Stage

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] **Step 1: Introduce semantic theme tokens with compatibility aliases**

At the start of `:root`, define:

```css
--night-950: #070a0e;
--night-900: #0b1015;
--night-850: #10161d;
--night-800: #151c24;
--night-700: #25303b;
--paper-100: #edf1f3;
--paper-300: #b7c0c8;
--paper-500: #7d8996;
--signal: #b1ef63;
--evidence: #8faaff;
--danger: #f06f73;
```

Map `--bg`, `--surface`, `--surface-soft`, `--surface-tint`, `--ink`, `--muted`, `--faint`, `--line`, `--line-strong`, `--green`, and `--blue` to those semantic values so untouched views continue rendering. Replace the thick offset shadow with a restrained ambient shadow.

- [ ] **Step 2: Refine the global shell without changing behavior hooks**

Keep `.sidebar`, every `[data-view-target]`, `#sessionList`, `#sidebarNewChatButton`, `#activeProjectLabel`, and `#healthStatus`. Reduce the desktop sidebar to about 228px; use fine rules, compact mono metadata, a signal-colored active indicator, and no duplicated oversized topbar brand. Add visible `:focus-visible` outlines for links, buttons, inputs, selects, and textareas.

- [ ] **Step 3: Replace only the contents of `#chatWelcome`**

Retain `#chatContextLabel` and create this stable structure:

```html
<div class="welcome-copy">
  <span class="welcome-kicker">RESEARCH CONVERSATION / 01</span>
  <h1>从一个问题开始研究。</h1>
  <p class="muted" id="chatContextLabel">Internal Research</p>
</div>
<div class="evidence-constellation" aria-hidden="true">
  <span class="constellation-orbit orbit-one"></span>
  <span class="constellation-orbit orbit-two"></span>
  <i class="constellation-node node-one"></i>
  <i class="constellation-node node-two"></i>
  <i class="constellation-node node-three"></i>
  <i class="constellation-node node-four"></i>
  <i class="constellation-node node-core"></i>
</div>
<div class="welcome-index" aria-label="当前研究空间状态">
  <span><strong>ASK</strong> Research question</span>
  <span><strong>TRACE</strong> Verifiable evidence</span>
  <span><strong>READ</strong> Editorial answer</span>
</div>
```

- [ ] **Step 4: Style the approved editorial composition and balanced motion**

Use a serif stack only for the welcome/answer display typography; keep controls in the local sans stack and metadata in a local mono stack. Animate constellation opacity/transform at low frequency. A body `.is-researching` state may briefly increase node brightness. Do not use remote fonts, images, canvas, packages, or runtime dependencies.

- [ ] **Step 5: Add explicit accessibility and responsive fallbacks**

At `max-width: 768px`, collapse the persistent session history and keep navigation/actions reachable. At `max-width: 480px`, scale the display title, lower constellation contrast, let composer actions wrap, and prevent horizontal overflow. Add:

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
  .evidence-constellation,
  .constellation-node,
  .constellation-orbit,
  .thinking-step {
    animation: none !important;
    transform: none !important;
  }
}
```

- [ ] **Step 6: Run the focused tests**

Run: `python -m pytest tests/test_static_frontend.py -q`  
Expected: theme/structure/motion tests pass; citation tests may remain red until Task 3.

### Task 3: Make Chat the Default and Render Safe Inline Footnotes

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] **Step 1: Change only the implicit startup route**

Use `var activeView = "chat";` and `setView((location.hash || "#chat").slice(1));`. Do not alter explicit hash navigation or button behavior.

- [ ] **Step 2: Add safe citation marker construction**

Implement this text-node-based shape; never assign answer content to `innerHTML`:

```javascript
function createCitationMarker(index, citation) {
  var marker = document.createElement("sup");
  marker.className = "citation-marker";
  marker.dataset.citationIndex = String(index);
  var url = citation ? getSourceUrl(citation) : null;
  var target = url ? document.createElement("a") : document.createElement("span");
  if (url) {
    target.href = url;
    target.target = "_blank";
    target.rel = "noopener";
  } else {
    target.tabIndex = 0;
  }
  target.textContent = String(index);
  target.setAttribute("aria-label", "查看来源 " + index);
  marker.appendChild(target);
  return marker;
}

function renderCitedText(text, citations, container) {
  container.replaceChildren();
  var value = String(text || "");
  var pattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  var cursor = 0;
  var match;
  while ((match = pattern.exec(value)) !== null) {
    container.appendChild(document.createTextNode(value.slice(cursor, match.index)));
    match[1].split(",").forEach(function (rawIndex) {
      var index = Number(rawIndex.trim());
      container.appendChild(createCitationMarker(index, (citations || [])[index - 1]));
    });
    cursor = pattern.lastIndex;
  }
  container.appendChild(document.createTextNode(value.slice(cursor)));
}
```

- [ ] **Step 3: Apply the safe renderer to current and historical answers**

In `showAnswer`, use `renderCitedText(text, citations, $("answerText"));`. In `appendMessage`, use `renderCitedText(content || "", citations, body);`. Keep user/tool content readable when markers or citation data are missing.

- [ ] **Step 4: Number the source list consistently**

Have `renderSources` build an ordered list and set `data-citation-index` on every source item. Keep title assignment through `textContent`, and retain `_blank` plus `noopener` for links.

- [ ] **Step 5: Couple request state to restrained visual feedback**

Add `setResearchingState(active)` that toggles `document.body.classList.toggle("is-researching", !!active)`. Set it immediately before Agent/RAG work and clear it in the existing submit promise `.finally()` path. Do not change payloads or error behavior.

- [ ] **Step 6: Run the full focused frontend suite**

Run: `python -m pytest tests/test_static_frontend.py -q`  
Expected: all focused tests pass.

### Task 4: Regression and Browser Acceptance

**Files:**
- Modify only if verification exposes a phase-one regression: `src/app/static/index.html`, `tests/test_static_frontend.py`

- [ ] **Step 1: Run static/startup regression tests**

Run: `python -m pytest tests/test_static_frontend.py tests/test_app_startup.py -q`  
Expected: all tests pass.

- [ ] **Step 2: Run the full repository suite**

Run: `python -m pytest -q`  
Expected: all tests pass. Report exact output for any unrelated environment-dependent failure; do not hide it.

- [ ] **Step 3: Verify JavaScript independently**

Run: `python -m pytest tests/test_static_frontend.py::test_inline_script_is_valid_javascript -q`  
Expected: `1 passed`.

- [ ] **Step 4: Check desktop, tablet, and mobile layouts**

At 1440×1000, 1024×768, and 390×844, verify the default chat stage, navigation, composer, and status are visible; `document.documentElement.scrollWidth <= window.innerWidth`; and capture screenshots.

- [ ] **Step 5: Smoke preserved interactions**

Verify default chat and explicit `#files`/`#runs`; new-chat clearing; attachment picker; Agent/RAG selection; Enter, Shift+Enter, and IME handling; safe `[1]` inline marker plus matching source; and reduced-motion behavior.

- [ ] **Step 6: Review scope and whitespace**

Run:

```powershell
git diff --check
git status --short
git diff -- src/app/static/index.html tests/test_static_frontend.py
```

Expected: no whitespace errors and no out-of-scope backend/deployment changes.

