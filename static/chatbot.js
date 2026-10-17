/**
 * MediNova Clinical AI Copilot — Frontend Controller (v2.0 Bulletproof)
 * Self-contained, context-aware clinical assistant for MediNova.
 * Features:
 * - Bulletproof initialization & DOM auto-injection
 * - Exposed global API: window.toggleMediNovaChat, window.openMediNovaChat, window.closeMediNovaChat, etc.
 * - Guaranteed visibility via explicit inline styles & class toggling
 * - Delegated document-level click listener for robust launcher response
 * - Live biomarker breakdown, symptom triage, medical explanations, form auto-fill
 * - Voice Input (Speech-to-Text) & Voice Readout (TTS)
 */

(function () {
    'use strict';

    // State management
    let isChatOpen = false;
    let isMaximized = false;
    let isSpeechEnabled = false;
    let isListening = false;
    let recognition = null;
    let speechSynth = window.speechSynthesis || null;

    // Detect Active Context based on DOM and URL
    function detectPageContext() {
        const path = window.location.pathname.toLowerCase();
        let pageName = 'Diagnostic Suite Overview';
        let condition = null;

        if (path.includes('heart')) {
            pageName = 'Cardiovascular Screening (/heart)';
            condition = 'Heart Disease';
        } else if (path.includes('diabetes')) {
            pageName = 'Diabetes Risk Screening (/diabetes)';
            condition = 'Diabetes';
        } else if (path.includes('kidney')) {
            pageName = 'Chronic Kidney Disease (/kidney)';
            condition = 'Kidney Disease';
        } else if (path.includes('liver')) {
            pageName = 'Hepatic Panel Screening (/liver)';
            condition = 'Liver Disease';
        } else if (path.includes('breast_cancer')) {
            pageName = 'Oncology Malignancy Screening (/breast_cancer)';
            condition = 'Breast Cancer';
        } else if (path.includes('malaria')) {
            pageName = 'Microscopic Cell Triage (/malaria)';
            condition = 'Malaria';
        } else if (path.includes('pneumonia')) {
            pageName = 'Chest Radiograph Analysis (/pneumonia)';
            condition = 'Pneumonia';
        } else if (path.includes('features')) {
            pageName = 'Diagnostic Modalities & Features';
        } else if (path.includes('contact')) {
            pageName = 'Emergency Directory & 24/7 Hotlines';
        } else if (path === '/' || path.includes('home')) {
            pageName = 'MediNova Diagnostic Hub';
        }

        // Check if there is an active screening outcome displayed on page
        const resultCard = document.querySelector('.result-card, .result-header, .alert-success, .alert-danger, [class*="result"]');
        let resultSnippet = null;
        if (resultCard) {
            const heading = resultCard.querySelector('h2, h3, h4, .prediction-title, strong');
            if (heading) {
                resultSnippet = heading.innerText.trim();
            }
        }

        let contextPillText = `📍 Context: ${pageName}`;
        if (resultSnippet) {
            contextPillText = `📍 Context: ${condition || 'Screening'} (${resultSnippet})`;
        }

        return {
            pageName: pageName,
            path: path,
            condition: condition,
            hasResult: !!resultSnippet,
            resultSnippet: resultSnippet,
            pillText: contextPillText
        };
    }

    // Injects the entire Chatbot DOM structure if missing
    function injectChatbotDOM() {
        if (document.getElementById('medinova-chat-root')) return;

        const root = document.createElement('div');
        root.id = 'medinova-chat-root';

        root.innerHTML = `
            <!-- Floating Action Button (FAB) -->
            <button id="medinova-chat-fab" class="medinova-chat-fab" type="button" aria-label="Open MediNova AI Copilot" title="Open MediNova AI Copilot">
                <span class="pulse-ring"></span>
                <span class="fab-icon">🤖</span>
                <span class="fab-badge">MediNova AI</span>
            </button>

            <!-- Expandable Copilot Drawer -->
            <div id="medinova-chat-widget" class="medinova-chat-widget" style="display: none; opacity: 0; visibility: hidden; pointer-events: none;" role="dialog" aria-modal="true" aria-labelledby="chat-title">
                <!-- Header -->
                <div class="chat-header">
                    <div class="chat-title-group">
                        <div class="chat-avatar">
                            <span class="avatar-icon">⚕️</span>
                            <span class="status-indicator-dot online" title="Copilot Online"></span>
                        </div>
                        <div>
                            <div id="chat-title" class="chat-brand-name">MediNova Clinical AI Copilot</div>
                            <div class="chat-status-sub">Online • Context-Aware Clinical Engine</div>
                        </div>
                    </div>
                    <div class="chat-header-actions">
                        <button id="btn-toggle-tts" class="chat-tool-btn" type="button" title="Toggle Voice Readout (TTS)">
                            <span id="tts-icon">🔇</span>
                        </button>
                        <button id="btn-maximize-chat" class="chat-tool-btn" type="button" title="Maximize / Dock View">
                            <span id="maximize-icon">⤢</span>
                        </button>
                        <button id="btn-clear-chat" class="chat-tool-btn" type="button" title="Clear Chat History">
                            <span>🗑️</span>
                        </button>
                        <button id="btn-close-chat" class="chat-tool-btn close-btn" type="button" title="Minimize Copilot">
                            <span>✕</span>
                        </button>
                    </div>
                </div>

                <!-- Active Context Banner -->
                <div id="chat-context-banner" class="chat-context-banner">
                    <span id="chat-context-text">📍 Context: Detecting Active Session...</span>
                </div>

                <!-- Messages Container -->
                <div id="chat-messages-container" class="chat-messages-container">
                    <!-- Welcome Message -->
                    <div class="chat-message assistant-message">
                        <div class="message-avatar">🤖</div>
                        <div class="message-bubble">
                            <div class="message-sender">MediNova Clinical Copilot</div>
                            <div class="message-body">
                                <p>Hello, I am your <strong>MediNova AI Clinical Copilot</strong>. I am integrated directly with MediNova's 7 diagnostic screening modalities, validated biomarker ranges, and clinical emergency protocols.</p>
                                <p>How can I assist your clinical evaluation today?</p>
                                <ul>
                                    <li>Ask to <strong>explain your test results</strong> and compare biomarkers.</li>
                                    <li>Describe patient symptoms for <strong>smart triage & tool routing</strong>.</li>
                                    <li>Provide vitals for <strong>conversational form auto-filling</strong>.</li>
                                    <li>Inquire about clinical targets (e.g. <em>"Normal Biomarker Reference Ranges"</em>).</li>
                                </ul>
                            </div>
                            <div class="message-meta">Zero-Key Clinical Knowledge Graph Ready</div>
                        </div>
                    </div>
                </div>

                <!-- Step-by-Step Thinking Indicator -->
                <div id="chat-thinking-indicator" class="chat-thinking-indicator" style="display: none;">
                    <div class="thinking-spinner"></div>
                    <span id="thinking-step-text">Analyzing clinical context & biomarkers...</span>
                </div>

                <!-- Quick-Action Suggestion Chips -->
                <div id="chat-followups-container" class="chat-followups-container">
                    <button type="button" class="chat-chip" data-query="Check my symptoms and recommend a MediNova screening test">🩺 Check Symptoms & Recommend a Test</button>
                    <button type="button" class="chat-chip" data-query="Explain my latest diagnostic test result and biomarker analysis">📊 Explain My Latest Test Result</button>
                    <button type="button" class="chat-chip" data-query="What are normal clinical biomarker reference ranges?">🧪 Normal Biomarker Reference Ranges</button>
                    <button type="button" class="chat-chip" data-query="What are acute red-flag emergency symptoms requiring immediate care?">🚨 Red-Flag Emergency Warning Signs</button>
                </div>

                <!-- Input Footer -->
                <div class="chat-footer">
                    <div class="chat-input-wrapper">
                        <textarea id="chat-input-field" placeholder="Ask MediNova Copilot or speak symptoms..." rows="1"></textarea>
                        <button id="btn-mic-input" class="chat-mic-btn" type="button" title="Voice Input (Speech to Text)">
                            <span id="mic-icon">🎙️</span>
                        </button>
                        <button id="btn-send-message" class="chat-send-btn" type="button" title="Send Message">
                            <span>➤</span>
                        </button>
                    </div>
                    <div class="chat-disclaimer">
                        <span>MediNova Copilot provides clinical decision support. Always consult a licensed physician for medical diagnosis.</span>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(root);
        bindChatEvents();
        updateContextBanner();
    }

    // Update Top Context Banner
    function updateContextBanner() {
        const ctx = detectPageContext();
        const bannerText = document.getElementById('chat-context-text');
        if (bannerText) {
            bannerText.innerText = ctx.pillText;
        }
    }

    // Markdown & Table Formatter
    function formatMarkdown(text) {
        if (!text) return '';

        let html = text;

        // Escape dangerous HTML entities
        html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Tables
        const lines = html.split('\n');
        let inTable = false;
        let tableHtml = [];
        let newLines = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('|') && line.endsWith('|')) {
                const cells = line.split('|').slice(1, -1).map(c => c.trim());
                if (cells.every(c => /^[-:\s]+$/.test(c))) {
                    continue; // table separator
                }
                if (!inTable) {
                    inTable = true;
                    tableHtml.push('<div class="copilot-table-responsive"><table class="copilot-biomarker-table"><thead><tr>');
                    cells.forEach(c => tableHtml.push(`<th>${c}</th>`));
                    tableHtml.push('</tr></thead><tbody>');
                } else {
                    tableHtml.push('<tr>');
                    cells.forEach((c) => {
                        let cellContent = c;
                        if (c.includes('⚠️ Elevated') || c.includes('Elevated') || c.includes('High Risk') || c.includes('Abnormal')) {
                            cellContent = `<span class="badge-abnormal">${c}</span>`;
                        } else if (c.includes('✅ Optimal') || c.includes('Optimal') || c.includes('Normal')) {
                            cellContent = `<span class="badge-normal">${c}</span>`;
                        }
                        tableHtml.push(`<td>${cellContent}</td>`);
                    });
                    tableHtml.push('</tr>');
                }
            } else {
                if (inTable) {
                    inTable = false;
                    tableHtml.push('</tbody></table></div>');
                    newLines.push(tableHtml.join(''));
                    tableHtml = [];
                }
                newLines.push(line);
            }
        }
        if (inTable) {
            tableHtml.push('</tbody></table></div>');
            newLines.push(tableHtml.join(''));
        }

        html = newLines.join('\n');

        // Headers
        html = html.replace(/^### (.*$)/gim, '<h4 class="copilot-h4">$1</h4>');
        html = html.replace(/^## (.*$)/gim, '<h3 class="copilot-h3">$1</h3>');
        html = html.replace(/^# (.*$)/gim, '<h2 class="copilot-h2">$1</h2>');

        // Bold & Italic
        html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code class="copilot-inline-code">$1</code>');

        // Emergency Callout Box
        html = html.replace(/🚨 \*\*HIGH-PRIORITY CLINICAL EMERGENCY ALERT:\*\*([\s\S]*?)(?=\n\n|$)/g, function (match, p1) {
            return `<div class="copilot-emergency-box">
                <div class="emergency-title">🚨 CRITICAL EMERGENCY ALERT</div>
                <div class="emergency-content">${p1.trim()}</div>
            </div>`;
        });

        // Lists
        html = html.replace(/^\s*[-•*]\s+(.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>(\s*<li>.*<\/li>)*)/gim, '<ul class="copilot-list">$1</ul>');

        // Line breaks
        html = html.replace(/\n\n+/g, '</p><p>');
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p><h([2-4])/g, '<h$1');
        html = html.replace(/<\/h([2-4])><\/p>/g, '</h$1>');
        html = html.replace(/<p><div/g, '<div');
        html = html.replace(/<\/div><\/p>/g, '</div>');
        html = html.replace(/<p><ul/g, '<ul');
        html = html.replace(/<\/ul><\/p>/g, '</ul>');
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    // Render Message Bubble
    function appendMessage(sender, content, options = {}) {
        const container = document.getElementById('chat-messages-container');
        if (!container) return;

        const isUser = sender === 'user';
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${isUser ? 'user-message' : 'assistant-message'}`;

        const avatar = isUser ? '👤' : '🤖';
        const senderName = isUser ? 'You' : 'MediNova Clinical Copilot';
        const badge = options.confidence_badge || (isUser ? '' : 'MediNova Clinical Intelligence');

        let actionsHtml = '';
        if (options.suggested_actions && options.suggested_actions.length > 0) {
            actionsHtml = '<div class="copilot-action-buttons">';
            options.suggested_actions.forEach(act => {
                if (act.url) {
                    actionsHtml += `<a href="${act.url}" class="btn-copilot-action" title="Navigate to ${act.label}">${act.label}</a>`;
                } else if (act.action === 'print') {
                    actionsHtml += `<button type="button" class="btn-copilot-action btn-print-action" onclick="window.print()">🖨️ Print Medical Report</button>`;
                }
            });
            actionsHtml += '</div>';
        }

        // Form Auto-fill Button
        let autoFillHtml = '';
        if (options.form_fill_data) {
            const dataStr = encodeURIComponent(JSON.stringify(options.form_fill_data));
            autoFillHtml = `
                <div class="copilot-autofill-prompt">
                    <button type="button" class="btn-autofill-action" data-autofill="${dataStr}">
                        ⚡ Auto-Fill Form with Extracted Biomarkers
                    </button>
                    <span class="autofill-hint">${Object.keys(options.form_fill_data).length} parameters identified</span>
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-bubble">
                <div class="message-sender">${senderName}</div>
                <div class="message-body">${isUser ? escapeHtml(content) : formatMarkdown(content)}</div>
                ${autoFillHtml}
                ${actionsHtml}
                ${badge ? `<div class="message-meta">${badge}</div>` : ''}
            </div>
        `;

        container.appendChild(msgDiv);
        container.scrollTop = container.scrollHeight;

        // Auto-fill button listener
        const fillBtn = msgDiv.querySelector('.btn-autofill-action');
        if (fillBtn) {
            fillBtn.addEventListener('click', function () {
                try {
                    const data = JSON.parse(decodeURIComponent(this.getAttribute('data-autofill')));
                    triggerFormAutoFill(data);
                    this.innerHTML = '✅ Form Populated Successfully!';
                    this.classList.add('applied');
                    this.disabled = true;
                } catch (e) {
                    console.error('Auto-fill error:', e);
                }
            });
        }

        // TTS Readout
        if (!isUser && isSpeechEnabled && speechSynth) {
            speakText(content);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.innerText = text;
        return div.innerHTML;
    }

    // Text-to-Speech (Voice Readout)
    function speakText(text) {
        if (!speechSynth) return;
        speechSynth.cancel();

        let clean = text.replace(/[*#`_~[\]()|]/g, ' ')
                        .replace(/[^\w\s.,?!'-]/g, ' ')
                        .replace(/\s+/g, ' ')
                        .trim();

        if (!clean) return;

        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        const voices = speechSynth.getVoices();
        const preferredVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Neural')));
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }

        speechSynth.speak(utterance);
    }

    // Step-by-Step Thinking Indicator Animation
    let thinkingInterval = null;
    function showThinking(show) {
        const el = document.getElementById('chat-thinking-indicator');
        const textEl = document.getElementById('thinking-step-text');
        if (!el || !textEl) return;

        if (show) {
            el.style.display = 'flex';
            const steps = [
                'Analyzing clinical context & biomarkers...',
                'Cross-referencing MediNova clinical protocols...',
                'Synthesizing diagnostic recommendations & references...'
            ];
            let stepIdx = 0;
            textEl.innerText = steps[stepIdx];
            clearInterval(thinkingInterval);
            thinkingInterval = setInterval(() => {
                stepIdx = (stepIdx + 1) % steps.length;
                textEl.innerText = steps[stepIdx];
            }, 850);
        } else {
            clearInterval(thinkingInterval);
            el.style.display = 'none';
        }
    }

    // Update Follow-Up Chips
    function updateFollowUpChips(chips) {
        const container = document.getElementById('chat-followups-container');
        if (!container) return;

        if (!chips || chips.length === 0) {
            chips = [
                'Check my symptoms and recommend a MediNova screening test',
                'Explain my latest diagnostic test result and biomarker analysis',
                'What are normal clinical biomarker reference ranges?',
                'What are acute red-flag emergency symptoms requiring immediate care?'
            ];
        }

        container.innerHTML = '';
        chips.forEach(query => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'chat-chip';
            btn.innerText = query;
            btn.setAttribute('data-query', query);
            btn.addEventListener('click', function () {
                const q = this.getAttribute('data-query') || this.innerText;
                sendMessage(q);
            });
            container.appendChild(btn);
        });
    }

    // Conversational Form Auto-Fill Implementation
    function triggerFormAutoFill(formData) {
        if (!formData || typeof formData !== 'object') return;

        let populatedCount = 0;
        let firstInput = null;

        for (const [key, val] of Object.entries(formData)) {
            const el = document.querySelector(`[name="${key}"], #${key}, [name="${key.toLowerCase()}"], #${key.toLowerCase()}`);
            if (el) {
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));

                el.classList.add('copilot-autofill-highlight');
                setTimeout(() => {
                    el.classList.remove('copilot-autofill-highlight');
                }, 2200);

                if (!firstInput) firstInput = el;
                populatedCount++;
            }
        }

        if (firstInput) {
            firstInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            firstInput.focus();
        } else {
            const submitBtn = document.querySelector('form button[type="submit"], button[type="submit"]');
            if (submitBtn) {
                submitBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }

    // Send Message to /api/chat
    async function sendMessage(messageText) {
        const inputField = document.getElementById('chat-input-field');
        const text = (messageText || (inputField ? inputField.value : '')).trim();
        if (!text) return;

        if (inputField) inputField.value = '';

        appendMessage('user', text);
        showThinking(true);

        const ctx = detectPageContext();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: text,
                    current_page: window.location.pathname,
                    page_context: {
                        condition: ctx.condition,
                        has_result: ctx.hasResult,
                        result_snippet: ctx.resultSnippet
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            showThinking(false);

            appendMessage('assistant', data.reply || 'No response generated.', {
                confidence_badge: data.confidence_badge,
                suggested_actions: data.suggested_actions,
                form_fill_data: data.form_fill_data,
                is_emergency: data.is_emergency
            });

            if (data.follow_up_chips) {
                updateFollowUpChips(data.follow_up_chips);
            }
        } catch (err) {
            showThinking(false);
            console.error('MediNova Copilot API error:', err);
            appendMessage('assistant', '⚠️ **MediNova Clinical Engine Notice:** Connection to diagnostic server was momentarily interrupted. Please verify your connection or try again.', {
                confidence_badge: 'System Notice'
            });
        }
    }

    // Clear Chat History
    async function clearChatHistory() {
        try {
            await fetch('/api/chat/clear', { method: 'POST' });
        } catch (e) {
            console.warn('Error calling clear endpoint:', e);
        }

        const container = document.getElementById('chat-messages-container');
        if (container) {
            container.innerHTML = `
                <div class="chat-message assistant-message">
                    <div class="message-avatar">🤖</div>
                    <div class="message-bubble">
                        <div class="message-sender">MediNova Clinical Copilot</div>
                        <div class="message-body">
                            <p>Diagnostic chat history cleared. Ready for new clinical analysis, biomarker reviews, or symptom routing.</p>
                        </div>
                        <div class="message-meta">Session Reset</div>
                    </div>
                </div>
            `;
        }
        updateFollowUpChips([
            'Check my symptoms and recommend a MediNova screening test',
            'Explain my latest diagnostic test result and biomarker analysis',
            'What are normal clinical biomarker reference ranges?',
            'What are acute red-flag emergency symptoms requiring immediate care?'
        ]);
    }

    // Voice Input Setup
    function setupVoiceInput() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            const micBtn = document.getElementById('btn-mic-input');
            if (micBtn) {
                micBtn.style.opacity = '0.4';
                micBtn.title = 'Speech-to-text not supported in this browser';
            }
            return;
        }

        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isListening = true;
            const micBtn = document.getElementById('btn-mic-input');
            if (micBtn) micBtn.classList.add('recording-pulse');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            const inputField = document.getElementById('chat-input-field');
            if (inputField) {
                inputField.value = transcript;
            }
            sendMessage(transcript);
        };

        recognition.onerror = (event) => {
            console.warn('Speech recognition error:', event.error);
            isListening = false;
            const micBtn = document.getElementById('btn-mic-input');
            if (micBtn) micBtn.classList.remove('recording-pulse');
        };

        recognition.onend = () => {
            isListening = false;
            const micBtn = document.getElementById('btn-mic-input');
            if (micBtn) micBtn.classList.remove('recording-pulse');
        };
    }

    // =========================================================================
    // GLOBAL WINDOW CONTROLS & EXPOSED HOOKS
    // =========================================================================

    function openMediNovaChat(initialPrompt) {
        injectChatbotDOM();

        isChatOpen = true;
        const widget = document.getElementById('medinova-chat-widget');
        const fab = document.getElementById('medinova-chat-fab');

        if (widget) {
            widget.classList.add('active', 'open');
            widget.style.display = 'flex';
            widget.style.opacity = '1';
            widget.style.visibility = 'visible';
            widget.style.pointerEvents = 'auto';
            widget.style.zIndex = '99999';
            widget.style.transform = 'translateY(0) scale(1)';

            updateContextBanner();

            const inputField = document.getElementById('chat-input-field');
            if (inputField) {
                setTimeout(() => inputField.focus(), 150);
            }
        }

        if (fab) {
            fab.classList.add('fab-active');
        }

        if (initialPrompt && typeof initialPrompt === 'string') {
            sendMessage(initialPrompt);
        }
    }

    function closeMediNovaChat() {
        isChatOpen = false;
        const widget = document.getElementById('medinova-chat-widget');
        const fab = document.getElementById('medinova-chat-fab');

        if (widget) {
            widget.classList.remove('active', 'open');
            widget.style.display = 'none';
            widget.style.opacity = '0';
            widget.style.visibility = 'hidden';
            widget.style.pointerEvents = 'none';
        }

        if (fab) {
            fab.classList.remove('fab-active');
        }
    }

    function toggleMediNovaChat() {
        if (isChatOpen) {
            closeMediNovaChat();
        } else {
            openMediNovaChat();
        }
    }

    // Expose Global API on window immediately
    window.toggleMediNovaChat = toggleMediNovaChat;
    window.openMediNovaChat = openMediNovaChat;
    window.closeMediNovaChat = closeMediNovaChat;

    window.askMediNovaExplain = function (promptText) {
        openMediNovaChat(promptText || "Please explain my latest diagnostic test results and biomarker analysis.");
    };

    window.askCopilotToExplainResult = function (conditionName, predictionText) {
        let summaryText = '';
        const rows = document.querySelectorAll('.table-biomarkers tbody tr, .report-summary-table tr, .patient-data-table tr, .table tr');
        if (rows && rows.length > 0) {
            const parts = [];
            rows.forEach(r => {
                const cols = r.querySelectorAll('td, th');
                if (cols.length >= 2) {
                    parts.push(`${cols[0].innerText.trim()}: ${cols[1].innerText.trim()}`);
                }
            });
            if (parts.length > 0) {
                summaryText = ` Key parameters: ${parts.slice(0, 8).join(', ')}.`;
            }
        }

        const prompt = `Please explain my ${conditionName || 'screening'} diagnostic result (${predictionText || 'Outcome Recorded'}). Analyze which submitted biomarkers were flagged, explain what clinical steps I should take next, and summarize recommended specialist consultations.${summaryText}`;
        openMediNovaChat(prompt);
    };

    // Bind Event Listeners
    function bindChatEvents() {
        const fab = document.getElementById('medinova-chat-fab');
        const closeBtn = document.getElementById('btn-close-chat');
        const maxBtn = document.getElementById('btn-maximize-chat');
        const maxIcon = document.getElementById('maximize-icon');
        const clearBtn = document.getElementById('btn-clear-chat');
        const ttsBtn = document.getElementById('btn-toggle-tts');
        const ttsIcon = document.getElementById('tts-icon');
        const sendBtn = document.getElementById('btn-send-message');
        const inputField = document.getElementById('chat-input-field');
        const micBtn = document.getElementById('btn-mic-input');
        const widget = document.getElementById('medinova-chat-widget');

        if (fab) {
            fab.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                toggleMediNovaChat();
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                closeMediNovaChat();
            });
        }

        // Maximize Toggle
        if (maxBtn) {
            maxBtn.addEventListener('click', (e) => {
                e.preventDefault();
                isMaximized = !isMaximized;
                if (widget) {
                    widget.classList.toggle('maximized', isMaximized);
                }
                if (maxIcon) {
                    maxIcon.innerText = isMaximized ? '🗗' : '⤢';
                }
            });
        }

        // Clear Chat
        if (clearBtn) {
            clearBtn.addEventListener('click', clearChatHistory);
        }

        // TTS Toggle
        if (ttsBtn) {
            ttsBtn.addEventListener('click', () => {
                isSpeechEnabled = !isSpeechEnabled;
                if (ttsIcon) ttsIcon.innerText = isSpeechEnabled ? '🔊' : '🔇';
                ttsBtn.title = isSpeechEnabled ? 'Mute Voice Readout' : 'Enable Voice Readout (TTS)';
                if (!isSpeechEnabled && speechSynth) speechSynth.cancel();
            });
        }

        // Send on Button Click
        if (sendBtn) {
            sendBtn.addEventListener('click', () => sendMessage());
        }

        // Send on Enter
        if (inputField) {
            inputField.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        // Mic Speech Input
        setupVoiceInput();
        if (micBtn) {
            micBtn.addEventListener('click', () => {
                if (!recognition) return;
                if (isListening) {
                    recognition.stop();
                } else {
                    try {
                        recognition.start();
                    } catch (e) {
                        console.warn('Recognition start error:', e);
                    }
                }
            });
        }

        // Follow-up Chips
        document.querySelectorAll('.chat-chip').forEach(btn => {
            btn.addEventListener('click', function () {
                const query = this.getAttribute('data-query') || this.innerText;
                sendMessage(query);
            });
        });
    }

    // Delegated Global Click Listener for bulletproof button responsiveness
    document.addEventListener('click', function (e) {
        // Any click on the floating button or its children
        const launcher = e.target.closest('#medinova-chat-fab, .medinova-chat-fab, [data-action="open-chat"], .open-chat-btn');
        if (launcher) {
            e.preventDefault();
            e.stopPropagation();
            toggleMediNovaChat();
            return;
        }

        // Any click on close button
        const closeBtn = e.target.closest('#btn-close-chat, .chat-close-btn');
        if (closeBtn) {
            e.preventDefault();
            e.stopPropagation();
            closeMediNovaChat();
            return;
        }
    });

    // Initialize immediately if DOM is ready, or on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectChatbotDOM);
    } else {
        injectChatbotDOM();
    }
})();
