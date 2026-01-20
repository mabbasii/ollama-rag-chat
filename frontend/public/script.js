const API_URL = '';
let chatHistory = [];
let currentModel = 'rag-knowledge-base';
let isProcessing = false;

marked.setOptions({
    breaks: true,
    gfm: true,
});

// DOM elements
const chatContainer = document.getElementById('chat-container');
const messagesContainer = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcome-screen');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const modelSelect = document.getElementById('model-select');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);

    modelSelect.addEventListener('change', (e) => {
        currentModel = e.target.value;
        chatHistory = []; // Clear conversation history when switching models
        addSystemMessage(`Switched to ${getModelDisplayName(currentModel)}`);
    });

    userInput.focus();
    checkConnection();
});

async function checkConnection() {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        if (data.status === 'ok') {
            console.log('Connected to backend');
        }
    } catch (error) {
        console.error('Cannot connect to backend');
        addSystemMessage('Warning: Cannot connect to backend. Check if server is running.');
    }
}

function getModelDisplayName(model) {
    const names = {
        'rag-knowledge-base': 'Knowledge Base (RAG)',
        'rag-documentation': 'Documentation (RAG)',
        'llama3.1:8b': 'Llama 3.1 8B',
        'mistral:7b': 'Mistral 7B',
        'codellama:7b': 'Code Llama 7B'
    };
    return names[model] || model;
}

function isRagModel() {
    return currentModel.startsWith('rag-');
}

async function sendMessage() {
    const message = userInput.value.trim();

    if (!message || isProcessing) return;

    userInput.value = '';
    userInput.style.height = 'auto';

    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }

    addMessage('user', message);
    chatHistory.push({ role: 'user', content: message });

    const assistantMessage = addMessage('assistant', '');
    const contentDiv = assistantMessage.querySelector('.flex-1');

    isProcessing = true;
    sendButton.disabled = true;
    userInput.disabled = true;

    try {
        // Determine endpoint based on model type
        let endpoint;
        if (currentModel === 'rag-knowledge-base') {
            endpoint = '/api/rag';
        } else if (currentModel === 'rag-documentation') {
            endpoint = '/api/rag-2';
        } else {
            endpoint = '/api/chat';
        }

        const requestBody = isRagModel()
            ? {
                message: message,
                history: chatHistory.slice(0, -1)
              }
            : {
                model: currentModel,
                message: message,
                history: chatHistory.slice(0, -1)
              };

        console.log(`Sending to: ${endpoint}`);
        console.log(`Model: ${currentModel}`);

        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let thinkingText = '';
        let contentText = '';
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);

                if (!line) continue;

                if (line.startsWith('data: ')) {
                    const data = line.substring(6).trim();

                    if (data === '[DONE]') {
                        break;
                    }

                    try {
                        const parsed = JSON.parse(data);

                        if (parsed.token) {
                            if (parsed.type === 'thinking') {
                                thinkingText += parsed.token;
                                displayThinkingResponse(contentDiv, thinkingText, contentText);

                            } else if (parsed.type === 'content') {
                                contentText += parsed.token;
                                displayThinkingResponse(contentDiv, thinkingText, contentText);
                            } else {
                                contentText += parsed.token;
                                contentDiv.innerHTML = marked.parse(contentText);
                                contentDiv.classList.add('markdown-content');
                            }

                            scrollToBottom();
                        }

                        if (parsed.error) {
                            contentDiv.innerHTML = `<div class="text-red-400">Error: ${parsed.error}</div>`;
                            if (isRagModel() && parsed.error.includes('RAG API')) {
                                contentDiv.innerHTML += `<div class="mt-2 text-white/60 text-sm">Tip: Make sure the Python RAG server is running</div>`;
                            }
                        }
                    } catch (e) {
                        console.error('JSON parse error:', e);
                        continue;
                    }
                }
            }
        }

        if (contentText || thinkingText) {
            const finalContent = contentText || thinkingText;
            chatHistory.push({ role: 'assistant', content: finalContent });
        }

    } catch (error) {
        console.error('Error:', error);

        if (isRagModel()) {
            contentDiv.innerHTML = `
                <div class="text-red-400 mb-2">Error: Failed to connect to RAG server</div>
                <div class="text-white/60 text-sm">
                    <p class="mb-2">Make sure the Python RAG server is running:</p>
                    <div class="bg-base-800 p-3 rounded font-mono text-xs">
                        cd rag-api<br>
                        source venv/bin/activate<br>
                        python3 rag_api.py
                    </div>
                </div>
            `;
        } else {
            contentDiv.textContent = 'Error: Failed to connect to server';
        }
    } finally {
        isProcessing = false;
        sendButton.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

function displayThinkingResponse(contentDiv, thinkingText, contentText) {
    let html = '';

    if (thinkingText) {
        if (contentText) {
            html += `
                <details class="thinking-section mb-4">
                    <summary class="cursor-pointer text-accent text-sm mb-2 flex items-center gap-2">
                        <span>Thinking process</span>
                        <span class="text-white/40 text-xs">(click to expand)</span>
                    </summary>
                    <div class="thinking-content text-white/50 text-sm pl-4 border-l-2 border-white/20 mt-2">
                        ${escapeHtml(thinkingText)}
                    </div>
                </details>
            `;
        } else {
            html = `
                <div class="thinking-indicator mb-4">
                    <div class="text-accent text-sm mb-2 flex items-center gap-2">
                        <div class="thinking-dots flex items-center gap-1">
                            <span>Thinking</span>
                            <span class="dot">.</span>
                            <span class="dot">.</span>
                            <span class="dot">.</span>
                        </div>
                    </div>
                    <div class="thinking-content text-white/50 text-sm pl-4 border-l-2 border-accent/30">
                        ${escapeHtml(thinkingText)}
                    </div>
                </div>
            `;
        }
    }

    if (contentText) {
        html += marked.parse(contentText);
    }

    contentDiv.innerHTML = html;
    contentDiv.classList.add('markdown-content');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `group w-full py-6 message-enter ${role === 'assistant' ? 'bg-base-800/30' : ''}`;

    const innerDiv = document.createElement('div');
    innerDiv.className = 'max-w-3xl mx-auto px-4 flex gap-4';

    const avatar = document.createElement('div');
    avatar.className = `flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
        role === 'user'
            ? 'bg-accent text-black'
            : 'bg-gradient-to-br from-accent/20 to-accent/5 text-accent'
    }`;

    if (role === 'user') {
        avatar.innerHTML = `<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>`;
    } else {
        if (isRagModel()) {
            avatar.innerHTML = `<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>`;
        } else {
            avatar.innerHTML = `<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>`;
        }
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'flex-1 text-white/90 leading-7';

    if (content) {
        if (role === 'user') {
            contentDiv.textContent = content;
        } else {
            contentDiv.innerHTML = marked.parse(content);
            contentDiv.classList.add('markdown-content');
        }
    }

    innerDiv.appendChild(avatar);
    innerDiv.appendChild(contentDiv);
    messageDiv.appendChild(innerDiv);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();
    return messageDiv;
}

function addSystemMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'w-full py-4';

    const innerDiv = document.createElement('div');
    innerDiv.className = 'max-w-3xl mx-auto px-4 text-center';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'inline-block px-3 py-1 bg-white/5 rounded-full text-xs text-white/50';
    contentDiv.textContent = content;

    innerDiv.appendChild(contentDiv);
    messageDiv.appendChild(innerDiv);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
