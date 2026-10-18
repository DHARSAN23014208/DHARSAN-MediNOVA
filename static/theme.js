/**
 * MediNova Intelligent Theme Engine (Dark & Light Mode)
 * Clinical Medical AI Platform
 * Developed by Dharsankumar
 */
(function () {
    'use strict';

    function getPreferredTheme() {
        try {
            var stored = localStorage.getItem('medinova_theme');
            if (stored === 'light' || stored === 'dark') return stored;
        } catch (e) {}
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 'light';
        }
        return 'dark';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        try {
            localStorage.setItem('medinova_theme', theme);
        } catch (e) {}
        updateToggleButtons(theme);
    }

    function updateToggleButtons(theme) {
        var btns = document.querySelectorAll('.theme-toggle-btn');
        btns.forEach(function (btn) {
            var label = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
            btn.setAttribute('aria-label', label);
            btn.setAttribute('title', label);
            var textSpan = btn.querySelector('.theme-mode-text');
            if (textSpan) {
                textSpan.textContent = theme === 'dark' ? 'Light' : 'Dark';
            }
            if (theme === 'light') {
                btn.classList.add('theme-is-light');
                btn.classList.remove('theme-is-dark');
            } else {
                btn.classList.add('theme-is-dark');
                btn.classList.remove('theme-is-light');
            }
        });
    }

    function createThemeButton(theme) {
        var btn = document.createElement('button');
        btn.id = 'themeToggleBtn';
        btn.type = 'button';
        btn.className = 'theme-toggle-btn ' + (theme === 'light' ? 'theme-is-light' : 'theme-is-dark');
        var label = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        btn.setAttribute('aria-label', label);
        btn.setAttribute('title', label);
        btn.innerHTML = [
            '<span class="theme-icon-sun" aria-hidden="true">',
            '  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">',
            '    <circle cx="12" cy="12" r="5"></circle>',
            '    <line x1="12" y1="1" x2="12" y2="3"></line>',
            '    <line x1="12" y1="21" x2="12" y2="23"></line>',
            '    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>',
            '    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>',
            '    <line x1="1" y1="12" x2="3" y2="12"></line>',
            '    <line x1="21" y1="12" x2="23" y2="12"></line>',
            '    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>',
            '    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>',
            '  </svg>',
            '</span>',
            '<span class="theme-icon-moon" aria-hidden="true">',
            '  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">',
            '    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>',
            '  </svg>',
            '</span>',
            '<span class="theme-mode-text">' + (theme === 'dark' ? 'Light' : 'Dark') + '</span>'
        ].join('');
        btn.addEventListener('click', handleToggleClick);
        return btn;
    }

    function handleToggleClick(e) {
        if (e) e.preventDefault();
        var current = document.documentElement.getAttribute('data-theme') || 'dark';
        var next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
    }

    function initTheme() {
        var theme = getPreferredTheme();
        applyTheme(theme);

        // Ensure theme toggle button exists in navbar on all pages
        var navLinks = document.querySelector('.nav-links');
        if (navLinks && !document.getElementById('themeToggleBtn')) {
            var btn = createThemeButton(theme);
            var emergencyBtn = navLinks.querySelector('.nav-emergency-btn');
            if (emergencyBtn) {
                navLinks.insertBefore(btn, emergencyBtn);
            } else {
                navLinks.appendChild(btn);
            }
        } else {
            document.querySelectorAll('.theme-toggle-btn').forEach(function (btn) {
                btn.removeEventListener('click', handleToggleClick);
                btn.addEventListener('click', handleToggleClick);
            });
        }
    }

    // Immediately execute attribute application to prevent flash
    try {
        var initT = getPreferredTheme();
        document.documentElement.setAttribute('data-theme', initT);
    } catch (e) {}

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }

    // Sync across browser tabs
    window.addEventListener('storage', function (e) {
        if (e.key === 'medinova_theme' && (e.newValue === 'light' || e.newValue === 'dark')) {
            applyTheme(e.newValue);
        }
    });

    // Listen to system theme preference changes if user hasn't set manual override
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function (e) {
            try {
                if (!localStorage.getItem('medinova_theme')) {
                    applyTheme(e.matches ? 'light' : 'dark');
                }
            } catch (err) {}
        });
    }
})();
