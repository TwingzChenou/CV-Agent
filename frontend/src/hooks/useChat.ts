
import { useState } from 'react';

export type Message = {
    role: 'user' | 'assistant';
    content: string;
    type?: 'tool_call';
};

export type ChatStatus = 'idle' | 'loading' | 'streaming';

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [status, setStatus] = useState<ChatStatus>('idle');
    const [currentTool, setCurrentTool] = useState<string | null>(null);

    const sendMessage = async (query: string) => {
        if (!query.trim()) return;

        // Add user message
        const userMessage: Message = { role: 'user', content: query };
        setMessages((prev) => [...prev, userMessage]);
        setStatus('loading');

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error('Failed to fetch response');
            }

            // Handle response - assuming JSON for now as per backend implementation
            // Future TODO: Handle SSE for streaming and tool calls
            const data = await response.json();

            const assistantMessage: Message = {
                role: 'assistant',
                content: data.response || data.text || "No response received."
            };

            setMessages((prev) => [...prev, assistantMessage]);
            setStatus('idle');

        } catch (error) {
            console.error("Chat error:", error);
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: "Sorry, something went wrong. Please check if the backend is running." }
            ]);
            setStatus('idle');
        }
    };

    return {
        messages,
        sendMessage,
        status,
        currentTool
    };
}
