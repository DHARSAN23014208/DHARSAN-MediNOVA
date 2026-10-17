/**
 * MediNova Clinical AI Advisor — Interactive Doctor Advisor & Situation-Handling Engine
 * Version: 3.0 (Self-contained, DOM & CSS auto-injected, high-specificity z-index: 2147483647)
 */

(function () {
    'use strict';

    // Guard against multiple initializations
    if (window.__MEDINOVA_CHATBOT_INITIALIZED__) return;
    window.__MEDINOVA_CHATBOT_INITIALIZED__ = true;

    // Inject styles
    function injectStyles() {
        if (document.getElementById('medinova-chatbot-styles')) return;

        const style = document.createElement('style');
        style.id = 'medinova-chatbot-styles';
        style.textContent = `
            /* High-specificity FAB Launcher */
            .mn-fab {
                position: fixed !important;
                bottom: 24px !important;
                right: 24px !important;
                z-index: 2147483647 !important;
                display: flex !important;
                align-items: center !important;
                gap: 10px !important;
                padding: 12px 20px !important;
                background: #0f172a !important;
                color: #f8fafc !important;
                border: 1.5px solid #06b6d4 !important;
                border-radius: 9999px !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                cursor: pointer !important;
                pointer-events: auto !important;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 16px rgba(6, 182, 212, 0.45) !important;
                transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease, border-color 0.2s ease !important;
                user-select: none !important;
                outline: none !important;
            }
            .mn-fab:hover {
                transform: translateY(-2px) scale(1.02) !important;
                box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.7), 0 0 24px rgba(6, 182, 212, 0.75) !important;
                border-color: #22d3ee !important;
            }
            .mn-fab:active {
                transform: translateY(0) scale(0.98) !important;
            }
            .mn-fab-dot {
                width: 9px !important;
                height: 9px !important;
                background-color: #10b981 !important;
                border-radius: 50% !important;
                box-shadow: 0 0 8px #10b981 !important;
                display: inline-block !important;
                animation: mn-pulse 2s infinite ease-in-out !important;
            }
            @keyframes mn-pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.25); opacity: 0.7; }
            }

            /* Glassmorphic Drawer Window */
            .mn-drawer {
                position: fixed !important;
                bottom: 90px !important;
                right: 24px !important;
                width: 390px !important;
                height: 560px !important;
                max-height: calc(100vh - 120px) !important;
                z-index: 2147483647 !important;
                background: #070b14 !important;
                border: 1.5px solid rgba(6, 182, 212, 0.35) !important;
                border-radius: 18px !important;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.85), 0 0 30px rgba(6, 182, 212, 0.2) !important;
                display: none;
                flex-direction: column !important;
                overflow: hidden !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
                color: #f8fafc !important;
                pointer-events: auto !important;
                transform: translateY(12px) !important;
                opacity: 0 !important;
                visibility: hidden !important;
                transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.22s ease !important;
            }
            .mn-drawer.mn-open {
                display: flex !important;
                flex-direction: column !important;
                opacity: 1 !important;
                visibility: visible !important;
                pointer-events: auto !important;
                transform: translateY(0) !important;
            }

            /* Header */
            .mn-header {
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                padding: 14px 16px !important;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
                border-bottom: 1px solid rgba(6, 182, 212, 0.2) !important;
            }
            .mn-header-info {
                display: flex !important;
                flex-direction: column !important;
                gap: 2px !important;
            }
            .mn-header-title {
                font-size: 14.5px !important;
                font-weight: 700 !important;
                color: #f8fafc !important;
                display: flex !important;
                align-items: center !important;
                gap: 6px !important;
                margin: 0 !important;
            }
            .mn-header-subtitle {
                font-size: 11.5px !important;
                color: #94a3b8 !important;
                margin: 0 !important;
            }
            .mn-header-actions {
                display: flex !important;
                align-items: center !important;
                gap: 6px !important;
            }
            .mn-btn-header {
                background: rgba(255, 255, 255, 0.08) !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                color: #cbd5e1 !important;
                border-radius: 8px !important;
                padding: 5px 9px !important;
                font-size: 12px !important;
                cursor: pointer !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.18s ease !important;
                outline: none !important;
            }
            .mn-btn-header:hover {
                background: rgba(6, 182, 212, 0.25) !important;
                color: #ffffff !important;
                border-color: #06b6d4 !important;
            }

            /* Messages Container */
            .mn-messages {
                flex: 1 !important;
                overflow-y: auto !important;
                padding: 16px !important;
                display: flex !important;
                flex-direction: column !important;
                gap: 14px !important;
                scroll-behavior: smooth !important;
            }
            .mn-messages::-webkit-scrollbar {
                width: 6px !important;
            }
            .mn-messages::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.18) !important;
                border-radius: 3px !important;
            }

            /* Chat Bubbles */
            .mn-msg {
                display: flex !important;
                flex-direction: column !important;
                max-width: 90% !important;
                animation: mn-fade-in 0.2s ease-out !important;
            }
            @keyframes mn-fade-in {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .mn-msg-user {
                align-self: flex-end !important;
                background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
                color: #ffffff !important;
                padding: 10px 14px !important;
                border-radius: 14px 14px 3px 14px !important;
                font-size: 13.5px !important;
                line-height: 1.45 !important;
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
                word-break: break-word !important;
            }
            .mn-msg-bot {
                align-self: flex-start !important;
                background: rgba(15, 23, 42, 0.9) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: #e2e8f0 !important;
                padding: 12px 14px !important;
                border-radius: 14px 14px 14px 3px !important;
                font-size: 13.5px !important;
                line-height: 1.5 !important;
                word-break: break-word !important;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
            }
            .mn-msg-bot h3, .mn-msg-bot h4 {
                color: #38bdf8 !important;
                margin: 6px 0 4px 0 !important;
                font-size: 14px !important;
                font-weight: 700 !important;
            }
            .mn-msg-bot p {
                margin: 0 0 6px 0 !important;
            }
            .mn-msg-bot p:last-child {
                margin-bottom: 0 !important;
            }
            .mn-msg-bot ul, .mn-msg-bot ol {
                margin: 4px 0 6px 18px !important;
                padding: 0 !important;
            }
            .mn-msg-bot li {
                margin-bottom: 4px !important;
            }
            .mn-msg-bot blockquote {
                border-left: 3px solid #06b6d4 !important;
                background: rgba(6, 182, 212, 0.08) !important;
                margin: 8px 0 !important;
                padding: 6px 10px !important;
                border-radius: 0 8px 8px 0 !important;
                font-size: 12.5px !important;
                color: #cbd5e1 !important;
            }
            .mn-msg-bot strong {
                color: #ffffff !important;
            }

            /* Quick Reply Chips */
            .mn-chips-container {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 6px !important;
                margin-top: 10px !important;
            }
            .mn-chip {
                background: rgba(15, 23, 42, 0.95) !important;
                border: 1px solid rgba(6, 182, 212, 0.5) !important;
                color: #38bdf8 !important;
                padding: 6px 12px !important;
                border-radius: 9999px !important;
                font-size: 12px !important;
                font-weight: 500 !important;
                cursor: pointer !important;
                transition: all 0.18s ease !important;
                user-select: none !important;
                outline: none !important;
                display: inline-flex !important;
                align-items: center !important;
            }
            .mn-chip:hover {
                background: rgba(6, 182, 212, 0.2) !important;
                color: #ffffff !important;
                border-color: #38bdf8 !important;
                transform: translateY(-1px) !important;
            }

            /* Action Buttons (Screening Links) */
            .mn-actions-container {
                display: flex !important;
                flex-direction: column !important;
                gap: 6px !important;
                margin-top: 10px !important;
            }
            .mn-action-btn {
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 6px !important;
                background: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%) !important;
                color: #ffffff !important;
                padding: 8px 14px !important;
                border-radius: 8px !important;
                font-size: 12.5px !important;
                font-weight: 600 !important;
                text-decoration: none !important;
                box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3) !important;
                transition: transform 0.18s ease, box-shadow 0.18s ease !important;
            }
            .mn-action-btn:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 16px rgba(6, 182, 212, 0.45) !important;
                color: #ffffff !important;
            }

            /* Typing Indicator */
            .mn-typing {
                align-self: flex-start !important;
                background: rgba(15, 23, 42, 0.8) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                padding: 10px 14px !important;
                border-radius: 14px 14px 14px 3px !important;
                display: none;
                align-items: center !important;
                gap: 8px !important;
                font-size: 12.5px !important;
                color: #94a3b8 !important;
            }
            .mn-typing.active {
                display: flex !important;
            }
            .mn-dots {
                display: flex !important;
                gap: 4px !important;
            }
            .mn-dot {
                width: 6px !important;
                height: 6px !important;
                background: #06b6d4 !important;
                border-radius: 50% !important;
                animation: mn-dot-bounce 1.4s infinite ease-in-out both !important;
            }
            .mn-dot:nth-child(1) { animation-delay: -0.32s !important; }
            .mn-dot:nth-child(2) { animation-delay: -0.16s !important; }
            @keyframes mn-dot-bounce {
                0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
                40% { transform: scale(1); opacity: 1; }
            }

            /* Footer & Input */
            .mn-footer {
                padding: 12px !important;
                background: #070b14 !important;
                border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
            }
            .mn-input-box {
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
                background: #0f172a !important;
                border: 1px solid rgba(6, 182, 212, 0.3) !important;
                border-radius: 12px !important;
                padding: 6px 10px !important;
                transition: border-color 0.2s ease !important;
            }
            .mn-input-box:focus-within {
                border-color: #06b6d4 !important;
                box-shadow: 0 0 10px rgba(6, 182, 212, 0.3) !important;
            }
            .mn-input {
                flex: 1 !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
                color: #f8fafc !important;
                font-size: 13.5px !important;
                font-family: inherit !important;
                padding: 4px !important;
            }
            .mn-input::placeholder {
                color: #64748b !important;
            }
            .mn-send-btn {
                background: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%) !important;
                border: none !important;
                color: #ffffff !important;
                width: 32px !important;
                height: 32px !important;
                border-radius: 8px !important;
                cursor: pointer !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                font-size: 13px !important;
                font-weight: bold !important;
                transition: transform 0.15s ease, opacity 0.15s ease !important;
                outline: none !important;
                flex-shrink: 0 !important;
            }
            .mn-send-btn:hover {
                transform: scale(1.06) !important;
            }
            .mn-send-btn:active {
                transform: scale(0.95) !important;
            }
            .mn-disclaimer {
                font-size: 10px !important;
                color: #64748b !important;
                text-align: center !important;
                margin-top: 6px !important;
                line-height: 1.3 !important;
            }

            /* Mobile & Print Responsiveness */
            @media (max-width: 480px) {
                .mn-drawer {
                    width: calc(100vw - 32px) !important;
                    right: 16px !important;
                    bottom: 84px !important;
                    height: calc(100vh - 110px) !important;
                }
                .mn-fab {
                    bottom: 16px !important;
                    right: 16px !important;
                    padding: 10px 16px !important;
                    font-size: 13px !important;
                }
            }
            @media print {
                .mn-fab, .mn-drawer, .medinova-chat-fab, .medinova-chat-widget, #medinova-chat-root {
                    display: none !important;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Markdown-like formatting helper
    function renderMarkdown(text) {
        if (!text) return '';
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Headings
        html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
        html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');

        // Bold & Italic
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Blockquotes
        html = html.replace(/^&gt; (.*$)/gim, '<blockquote>$1</blockquote>');

        // Unordered lists
        html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>');
        html = html.replace(/<\/ul>\s*<ul>/g, '');

        // Paragraph breaks
        html = html.replace(/\n{2,}/g, '</p><p>');
        html = html.replace(/\n/g, '<br/>');

        return `<p>${html}</p>`;
    }

    // Build the chatbot DOM elements
    let drawerEl = null;
    let fabEl = null;
    let messagesEl = null;
    let typingEl = null;
    let inputEl = null;

    const INITIAL_GREETING = "Hi! I'm your **MediNova Clinical AI Advisor**. Tell me how you're feeling or pick a situation below—I'll ask a couple of quick clinical questions and guide you step-by-step on how to handle it safely.";
    const INITIAL_CHIPS = [
        "🫀 Chest pain or high BP",
        "🩸 High blood sugar / Diabetes advice",
        "🌡️ Fever, chills, or persistent cough",
        "🧪 Kidney or Liver symptom check",
        "📊 Explain my latest test result"
    ];

    function buildChatbotDOM() {
        // Remove old duplicate roots or buttons if present
        const oldRoot = document.getElementById('medinova-chat-root');
        if (oldRoot) oldRoot.remove();
        const oldFab = document.getElementById('medinova-chat-fab');
        if (oldFab) oldFab.remove();
        const oldWidget = document.getElementById('medinova-chat-widget');
        if (oldWidget) oldWidget.remove();

        // 1. Create working FAB
        fabEl = document.createElement('button');
        fabEl.className = 'mn-fab';
        fabEl.setAttribute('type', 'button');
        fabEl.setAttribute('aria-label', 'Open MediNova AI Doctor Advisor');
        fabEl.innerHTML = `
            <span class="mn-fab-dot"></span>
            <span>🤖 MediNova AI</span>
        `;
        fabEl.addEventListener('click', function (e) {
            e.stopPropagation();
            toggleMediNovaChat();
        });

        // 2. Create Drawer
        drawerEl = document.createElement('div');
        drawerEl.className = 'mn-drawer';
        drawerEl.setAttribute('role', 'dialog');
        drawerEl.setAttribute('aria-label', 'MediNova Clinical AI Advisor');
        drawerEl.innerHTML = `
            <div class="mn-header">
                <div class="mn-header-info">
                    <div class="mn-header-title">🩺 MediNova AI Advisor</div>
                    <div class="mn-header-subtitle">Interactive Symptom & Situation Guide</div>
                </div>
                <div class="mn-header-actions">
                    <button class="mn-btn-header mn-reset-btn" type="button" title="Reset consultation">↺ Reset</button>
                    <button class="mn-btn-header mn-close-btn" type="button" title="Close window">✕</button>
                </div>
            </div>

            <div class="mn-messages" id="mn-messages"></div>

            <div class="mn-typing" id="mn-typing">
                <div class="mn-dots">
                    <div class="mn-dot"></div>
                    <div class="mn-dot"></div>
                    <div class="mn-dot"></div>
                </div>
                <span>MediNova AI is analyzing...</span>
            </div>

            <div class="mn-footer">
                <div class="mn-input-box">
                    <input class="mn-input" id="mn-input-field" type="text" placeholder="Describe symptoms or reply..." autocomplete="off" />
                    <button class="mn-send-btn" id="mn-send-button" type="button" title="Send Message">➤</button>
                </div>
                <div class="mn-disclaimer">
                    MediNova provides preliminary clinical risk support. For medical emergencies, call 112 / 911 / 108.
                </div>
            </div>
        `;

        document.body.appendChild(fabEl);
        document.body.appendChild(drawerEl);

        messagesEl = drawerEl.querySelector('#mn-messages');
        typingEl = drawerEl.querySelector('#mn-typing');
        inputEl = drawerEl.querySelector('#mn-input-field');

        // Header event listeners
        drawerEl.querySelector('.mn-close-btn').addEventListener('click', closeMediNovaChat);
        drawerEl.querySelector('.mn-reset-btn').addEventListener('click', resetMediNovaChat);

        // Input send listeners
        drawerEl.querySelector('#mn-send-button').addEventListener('click', handleUserSend);
        inputEl.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleUserSend();
            }
        });

        // Insert initial message and quick replies
        appendBotMessage(INITIAL_GREETING, INITIAL_CHIPS, []);
    }

    function toggleMediNovaChat() {
        if (!drawerEl) return;
        const isOpen = drawerEl.classList.contains('mn-open');
        if (isOpen) {
            closeMediNovaChat();
        } else {
            openMediNovaChat();
        }
    }

    function openMediNovaChat(initialMessage) {
        if (!drawerEl) return;
        drawerEl.classList.add('mn-open');
        if (inputEl) {
            setTimeout(() => inputEl.focus(), 150);
        }
        if (initialMessage && typeof initialMessage === 'string' && initialMessage.trim()) {
            sendMessageToAPI(initialMessage.trim());
        }
    }

    function closeMediNovaChat() {
        if (!drawerEl) return;
        drawerEl.classList.remove('mn-open');
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'mn-msg mn-msg-user';
        msgDiv.textContent = text;
        messagesEl.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendBotMessage(replyMarkdown, quickReplies, actions) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'mn-msg mn-msg-bot';
        msgDiv.innerHTML = renderMarkdown(replyMarkdown);

        // Render Action buttons (e.g. Open Heart Screening)
        if (actions && actions.length > 0) {
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'mn-actions-container';
            actions.forEach(act => {
                if (act.url) {
                    const a = document.createElement('a');
                    a.className = 'mn-action-btn';
                    a.href = act.url;
                    a.textContent = act.label || 'Open Diagnostic Suite';
                    actionsContainer.appendChild(a);
                }
            });
            msgDiv.appendChild(actionsContainer);
        }

        // Render Quick Reply chips
        if (quickReplies && quickReplies.length > 0) {
            const chipsContainer = document.createElement('div');
            chipsContainer.className = 'mn-chips-container';
            quickReplies.forEach(chipText => {
                const btn = document.createElement('button');
                btn.className = 'mn-chip';
                btn.type = 'button';
                btn.textContent = chipText;
                btn.addEventListener('click', function () {
                    appendUserMessage(chipText);
                    sendMessageToAPI(chipText);
                });
                chipsContainer.appendChild(btn);
            });
            msgDiv.appendChild(chipsContainer);
        }

        messagesEl.appendChild(msgDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        if (!messagesEl) return;
        setTimeout(() => {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }, 50);
    }

    function showTyping() {
        if (typingEl) typingEl.classList.add('active');
        scrollToBottom();
    }

    function hideTyping() {
        if (typingEl) typingEl.classList.remove('active');
    }

    function handleUserSend() {
        if (!inputEl) return;
        const text = inputEl.value.trim();
        if (!text) return;
        inputEl.value = '';
        appendUserMessage(text);
        sendMessageToAPI(text);
    }

    function sendMessageToAPI(text) {
        showTyping();
        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                current_page: window.location.pathname
            })
        })
        .then(res => res.json())
        .then(data => {
            hideTyping();
            const reply = data.reply || "I am analyzing your situation. How can I further assist you?";
            const chips = data.quick_replies || data.follow_up_chips || [];
            const actions = data.actions || data.suggested_actions || [];
            appendBotMessage(reply, chips, actions);
        })
        .catch(err => {
            hideTyping();
            console.error('MediNova Chat Error:', err);
            appendBotMessage(
                "I encountered a temporary connection issue. Please check your network and try again, or consult local emergency services (**112 / 911 / 108**) if experiencing an acute medical emergency.",
                INITIAL_CHIPS,
                []
            );
        });
    }

    function resetMediNovaChat() {
        showTyping();
        fetch('/api/chat/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(res => res.json())
        .then(data => {
            hideTyping();
            messagesEl.innerHTML = '';
            const reply = data.reply || INITIAL_GREETING;
            const chips = data.quick_replies || INITIAL_CHIPS;
            const actions = data.actions || [];
            appendBotMessage(reply, chips, actions);
        })
        .catch(() => {
            hideTyping();
            messagesEl.innerHTML = '';
            appendBotMessage(INITIAL_GREETING, INITIAL_CHIPS, []);
        });
    }

    // Expose global methods
    window.toggleMediNovaChat = toggleMediNovaChat;
    window.openMediNovaChat = openMediNovaChat;
    window.closeMediNovaChat = closeMediNovaChat;
    window.resetMediNovaChat = resetMediNovaChat;
    window.askCopilotToExplainResult = function (condition, prediction) {
        openMediNovaChat("Explain my latest test result");
    };

    // Bind any static page buttons (e.g. .btn-copilot-explain or hero buttons)
    function bindStaticButtons() {
        document.querySelectorAll('.btn-copilot-explain, [onclick*="askCopilotToExplainResult"], [onclick*="toggleMediNovaChat"]').forEach(btn => {
            btn.onclick = function (e) {
                e.preventDefault();
                openMediNovaChat("Explain my latest test result");
            };
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            injectStyles();
            buildChatbotDOM();
            bindStaticButtons();
        });
    } else {
        injectStyles();
        buildChatbotDOM();
        bindStaticButtons();
    }
})();
