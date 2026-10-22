# MediNova Ekon-Inspired UI Kit

A lightweight, copy-in design starter for the existing MediNova Flask/Jinja2 app. It implements an Ekon-inspired visual direction: dark editorial styling, high-contrast typography, responsive cards, subtle hover transitions, marquee motion, scroll reveals, counter animation, and system-aware light/dark themes with a manual toggle.

This is an original implementation based on publicly described/visible design characteristics. It is **not** extracted Framer source code, and it does not include Ekon's original licensed layouts, CMS, images, or project file.

## Files

- `medinova-ekon.css` — design tokens, light/dark theme variables, responsive component styles, and motion.
- `medinova-ui.js` — theme switching, system theme detection, scroll reveal, count-up animation, optional mobile menu behavior, and optional scroll progress.
- `demo.html` — standalone preview using synthetic content only.
- `ANTIGRAVITY_IMPLEMENTATION_PROMPT.md` — ready-to-paste instructions for integrating this kit into MediNova safely.

## Quick install in the existing Flask project

1. Copy `medinova-ekon.css` to `static/css/medinova-ekon.css`.
2. Copy `medinova-ui.js` to `static/js/medinova-ui.js`.
3. In the shared Jinja base template (commonly `templates/base.html`), add this inside `<head>`:

   ```html
   <link rel="stylesheet" href="{{ url_for('static', filename='css/medinova-ekon.css') }}">
   ```

4. Add this before `</body>`:

   ```html
   <script src="{{ url_for('static', filename='js/medinova-ui.js') }}" defer></script>
   ```

5. Add a theme control where it fits the existing navigation:

   ```html
   <button class="mn-theme-toggle" type="button" data-theme-toggle aria-label="Switch theme" title="Switch theme">
     <span data-theme-icon aria-hidden="true"></span>
   </button>
   <button type="button" data-theme-system hidden>Use device theme</button>
   ```

6. Use the `mn-` classes gradually on existing UI components. Do not replace existing forms or routes blindly. Check the prompt file first.

## Theme behavior

- On first visit, theme follows the device/OS `prefers-color-scheme` setting.
- Clicking `[data-theme-toggle]` switches light/dark and saves the explicit choice in local storage.
- Clicking `[data-theme-system]` clears the saved override and resumes automatic device theme.
- Palette changes apply through CSS variables; markup/content does not have to be duplicated.
- Ensure the theme toggle is present only once per navigation area or reused consistently.

## Useful classes

- Layout: `.mn-container`, `.mn-section`, `.mn-section--compact`
- Type: `.mn-eyebrow`, `.mn-display`, `.mn-h1`, `.mn-h2`, `.mn-h3`, `.mn-lead`
- Navigation: `.mn-header`, `.mn-nav`, `.mn-brand`, `.mn-nav-links`, `.mn-theme-toggle`
- Buttons: `.mn-button`, `.mn-button--primary`, `.mn-button--small`
- Grids/cards: `.mn-grid`, `.mn-grid--2`, `.mn-grid--3`, `.mn-grid--4`, `.mn-card`
- Reveal motion: add `.mn-reveal` to an element
- Count up: add `data-count="7"` and optionally `data-count-suffix=" modules"` to an element
- Marquee: use `.mn-marquee` + `.mn-marquee-track`, with track contents duplicated once for a seamless loop
- Form styling: `.mn-form`, `.mn-form-grid`, `.mn-field`, `.mn-label`, `.mn-input`, `.mn-select`, `.mn-textarea`
- Results: `.mn-result`, `.mn-result-value`, `.mn-badge`

## Safety notes

- Do not apply a global CSS class migration before inspecting the existing templates.
- Existing class names and Flask routes should remain intact unless there is a tested reason to change them.
- Use only real prediction values, hospital data, ratings, and integrations from existing application logic.
- The `mn-` classes are optional styling hooks. They should be applied intentionally to avoid changing unrelated pages.
- Keep patient data, credentials, and private configuration out of this kit and out of client-side JavaScript.
