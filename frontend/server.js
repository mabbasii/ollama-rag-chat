const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// API endpoints
const OLLAMA_API = process.env.OLLAMA_API || 'http://localhost:11434/api';
const RAG_API_1 = process.env.RAG_API_1 || 'http://localhost:8001/api';  // First RAG endpoint
const RAG_API_2 = process.env.RAG_API_2 || 'http://localhost:8002/api';  // Second RAG endpoint

// ============================================================================
// DIRECT OLLAMA CHAT (for any Ollama model)
// ============================================================================
app.post('/api/chat', async (req, res) => {
    const { model, message, history } = req.body;

    try {
        // Format messages for Ollama
        const messages = [
            ...history.map(msg => ({
                role: msg.role,
                content: msg.content
            })),
            { role: 'user', content: message }
        ];

        // Set headers for Server-Sent Events (streaming)
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        const response = await fetch(`${OLLAMA_API}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                messages: messages,
                stream: true
            })
        });

        // Stream the response chunk by chunk
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                res.write('data: [DONE]\n\n');
                res.end();
                break;
            }

            // Decode the chunk
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
                try {
                    const data = JSON.parse(line);

                    if (data.message) {
                        // Extract BOTH content and thinking (for models that support it)
                        const content = data.message.content || '';
                        const thinking = data.message.thinking || '';

                        // Send whichever field has data
                        if (thinking) {
                            res.write(`data: ${JSON.stringify({ token: thinking, type: 'thinking' })}\n\n`);
                        } else if (content) {
                            res.write(`data: ${JSON.stringify({ token: content, type: 'content' })}\n\n`);
                        }
                    }

                    if (data.done) {
                        res.write('data: [DONE]\n\n');
                        res.end();
                        return;
                    }
                } catch (e) {
                    continue;
                }
            }
        }

    } catch (error) {
        console.error('Ollama Error:', error);
        res.write(`data: ${JSON.stringify({ error: 'Failed to communicate with Ollama' })}\n\n`);
        res.end();
    }
});

// ============================================================================
// RAG ENDPOINT 1 (e.g., Knowledge Base, Documentation)
// ============================================================================
app.post('/api/rag', async (req, res) => {
    const { message, history } = req.body;

    console.log('RAG Query:', message);

    try {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        const response = await fetch(`${RAG_API_1}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: history || [],
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`RAG API returned ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                res.write('data: [DONE]\n\n');
                res.end();
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            res.write(chunk);
        }

    } catch (error) {
        console.error('RAG API Error:', error);
        res.write(`data: ${JSON.stringify({ error: 'Failed to communicate with RAG API. Make sure Python server is running on port 8001.' })}\n\n`);
        res.end();
    }
});

// ============================================================================
// RAG ENDPOINT 2 (e.g., Second Knowledge Base)
// ============================================================================
app.post('/api/rag-2', async (req, res) => {
    const { message, history } = req.body;

    console.log('RAG-2 Query:', message);

    try {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        const response = await fetch(`${RAG_API_2}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: history || [],
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`RAG API returned ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                res.write('data: [DONE]\n\n');
                res.end();
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            res.write(chunk);
        }

    } catch (error) {
        console.error('RAG-2 API Error:', error);
        res.write(`data: ${JSON.stringify({ error: 'Failed to communicate with RAG API. Make sure Python server is running on port 8002.' })}\n\n`);
        res.end();
    }
});

// ============================================================================
// UTILITY ENDPOINTS
// ============================================================================

// Get available Ollama models
app.get('/api/models', async (req, res) => {
    try {
        const response = await fetch(`${OLLAMA_API}/tags`);
        const data = await response.json();
        res.json({
            success: true,
            models: data.models.map(m => m.name)
        });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to fetch models'
        });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Serve index.html for root
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('Ollama RAG Chat Server');
    console.log('='.repeat(60));
    console.log(`Server:      http://localhost:${PORT}`);
    console.log(`Ollama:      ${OLLAMA_API}`);
    console.log(`RAG API 1:   ${RAG_API_1}`);
    console.log(`RAG API 2:   ${RAG_API_2}`);
    console.log(`Streaming:   Enabled`);
    console.log('='.repeat(60) + '\n');
});
