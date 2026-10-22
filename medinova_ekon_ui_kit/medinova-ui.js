/* MediNova UI interactions: theme mode, reveal motion, count-up, mobile navigation. */
(() => {
  'use strict';

  const root = document.documentElement;
  const storageKey = 'medinova-theme';
  const themeQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
  const reduceMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false;

  function systemTheme() {
    return themeQuery && themeQuery.matches ? 'light' : 'dark';
  }

  function getSavedTheme() {
    try {
      const value = localStorage.getItem(storageKey);
      return value === 'light' || value === 'dark' ? value : null;
    } catch (_) {
      return null;
    }
  }

  function setTheme(theme, mode = 'manual') {
    const safeTheme = theme === 'light' ? 'light' : 'dark';
    root.setAttribute('data-theme', safeTheme);
    root.setAttribute('data-theme-mode', mode);
    updateThemeControls(safeTheme, mode);
  }

  function updateThemeControls(theme, mode) {
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
      const target = theme === 'dark' ? 'light' : 'dark';
      button.setAttribute('aria-label', `Switch to ${target} theme`);
      button.setAttribute('title', `Switch to ${target} theme`);
      button.setAttribute('aria-pressed', String(mode === 'manual'));
      const icon = button.querySelector('[data-theme-icon]');
      if (icon) {
        icon.innerHTML = theme === 'dark'
          ? '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="4" stroke="currentColor" stroke-width="1.7"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>'
          : '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20.2 15.4A8.5 8.5 0 0 1 8.6 3.8 8.7 8.7 0 1 0 20.2 15.4Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>';
      }
    });
    document.querySelectorAll('[data-theme-system]').forEach((button) => {
      button.hidden = mode !== 'manual';
      button.setAttribute('aria-label', 'Use device theme setting');
    });
  }

  // Apply saved choice early; otherwise follow the operating system automatically.
  const savedTheme = getSavedTheme();
  setTheme(savedTheme || systemTheme(), savedTheme ? 'manual' : 'system');

  document.addEventListener('click', (event) => {
    const toggle = event.target.closest('[data-theme-toggle]');
    if (toggle) {
      const nextTheme = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      setTheme(nextTheme, 'manual');
      try { localStorage.setItem(storageKey, nextTheme); } catch (_) { /* storage may be disabled */ }
      return;
    }

    const useSystem = event.target.closest('[data-theme-system]');
    if (useSystem) {
      try { localStorage.removeItem(storageKey); } catch (_) { /* storage may be disabled */ }
      setTheme(systemTheme(), 'system');
      return;
    }

    const menuButton = event.target.closest('[data-mobile-menu-toggle]');
    if (menuButton) {
      const menu = document.querySelector(menuButton.getAttribute('aria-controls') ? `#${CSS.escape(menuButton.getAttribute('aria-controls'))}` : '[data-mobile-menu]');
      if (menu) {
        const isOpen = menu.classList.toggle('is-open');
        menuButton.setAttribute('aria-expanded', String(isOpen));
      }
    }
  });

  if (themeQuery && typeof themeQuery.addEventListener === 'function') {
    themeQuery.addEventListener('change', () => {
      if (!getSavedTheme()) setTheme(systemTheme(), 'system');
    });
  } else if (themeQuery && typeof themeQuery.addListener === 'function') {
    themeQuery.addListener(() => {
      if (!getSavedTheme()) setTheme(systemTheme(), 'system');
    });
  }

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      document.querySelectorAll('[data-mobile-menu].is-open').forEach((menu) => menu.classList.remove('is-open'));
      document.querySelectorAll('[data-mobile-menu-toggle][aria-expanded="true"]').forEach((button) => button.setAttribute('aria-expanded', 'false'));
    }
  });

  // Reveal elements when they enter the viewport. Content remains visible if observer isn't supported.
  const revealItems = document.querySelectorAll('.mn-reveal');
  if (!reduceMotion && 'IntersectionObserver' in window && revealItems.length) {
    document.body.classList.add('mn-motion-ready');
    const observer = new IntersectionObserver((entries, currentObserver) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          currentObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -35px 0px' });
    revealItems.forEach((item) => observer.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add('is-visible'));
  }

  // Optional count-up elements: <span data-count="7" data-count-suffix=" modules">7 modules</span>.
  const counters = document.querySelectorAll('[data-count]');
  function formatCounter(element, number) {
    const decimals = Number(element.dataset.countDecimals || 0);
    const formatted = number.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    return `${element.dataset.countPrefix || ''}${formatted}${element.dataset.countSuffix || ''}`;
  }
  function animateCounter(element) {
    const target = Number(element.dataset.count);
    if (!Number.isFinite(target)) return;
    if (reduceMotion || !('requestAnimationFrame' in window)) {
      element.textContent = formatCounter(element, target);
      return;
    }
    const duration = Math.max(350, Math.min(Number(element.dataset.countDuration || 1100), 3000));
    const start = performance.now();
    const from = Number(element.dataset.countFrom || 0);
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      element.textContent = formatCounter(element, from + (target - from) * eased);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  if (counters.length) {
    if ('IntersectionObserver' in window && !reduceMotion) {
      const counterObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.5 });
      counters.forEach((counter) => counterObserver.observe(counter));
    } else {
      counters.forEach(animateCounter);
    }
  }

  // Optional reading progress line: place <div class="mn-progress" data-scroll-progress></div> in the body.
  const progressBar = document.querySelector('[data-scroll-progress]');
  if (progressBar) {
    let ticking = false;
    const updateProgress = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const progress = scrollable > 0 ? window.scrollY / scrollable : 0;
      progressBar.style.transform = `scaleX(${Math.max(0, Math.min(progress, 1))})`;
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) {
        window.requestAnimationFrame(updateProgress);
        ticking = true;
      }
    }, { passive: true });
    updateProgress();
  }
})();
