# Antigravity Prompt — Integrate the MediNova Ekon-Inspired UI Kit

You are working on my existing MediNova Flask healthcare application. Integrate the accompanying `medinova-ekon.css` and `medinova-ui.js` into the existing project safely.

Design reference: https://www.framer.com/marketplace/templates/ekon/
Live preview: https://ekon.framer.ai/

## Important context

This kit is an original, Ekon-inspired implementation, not the proprietary source code of the Framer template. Use it as a design system and interaction starter, then adapt it to the existing MediNova interface.

## Rules

1. First inspect the repository, current Git status, current templates, CSS, JavaScript, Flask routes, and all existing functionality.
2. Do not create a separate app, duplicate the project, reset Git, or overwrite work that is already valid.
3. Copy the CSS file into `static/css/medinova-ekon.css` and the JS file into `static/js/medinova-ui.js`, creating directories only when needed.
4. Import the CSS and JS through the shared Jinja base template if one exists. If the project uses a different shared layout, integrate through the correct existing entry point.
5. Add an accessible theme switcher. First visit must follow `prefers-color-scheme`; manual light/dark selection must persist; an optional “Use device theme” control must restore automatic behavior.
6. Apply the `mn-` styles incrementally to the shared navigation, home page, cards, forms, results, hospital finder, lab scanner, and appointment UI. Reuse existing design components where appropriate.
7. Preserve all seven routes: `/diabetes`, `/heart`, `/kidney`, `/liver`, `/breast_cancer`, `/malaria`, `/pneumonia`.
8. Preserve existing Flask endpoints, field names, payloads, model inference, scanner extraction, geocoding and location fallback, specialty filtering, appointment workflow, Google Sheets logging, environment variables, and Docker's 512 MB constraint.
9. Use `.mn-reveal` only on elements where scroll-reveal motion improves clarity. Use `data-count` only for truthful, verified metrics; do not invent healthcare statistics. Use marquees sparingly and pause them on hover. Respect reduced-motion settings.
10. Maintain responsive layouts and accessible keyboard focus, labels, color contrast, and readable medical content in both themes.
11. Do not use placeholders to fake successful predictions, hospital search, appointment booking, or logging.
12. Do not change prediction or backend business logic solely for visual reasons.

## Required process

- Audit first and identify the correct integration points.
- Integrate the assets and add the theme control.
- Apply the styles to the existing pages in manageable steps.
- Test the app startup and all seven diagnostic routes.
- Check the scanner, hospital search (including geolocation failure and manual location entry), appointments, and Google Sheets logging where credentials permit.
- Inspect the UI in light and dark modes, at desktop/tablet/mobile widths.
- Fix regressions and verify browser console and Flask logs.

## Final response

Report what was changed, exact file paths, route/feature test results, untested areas, and how to run the project. Clearly distinguish tests actually executed from those that could not be performed. Do not commit or push without my approval.
