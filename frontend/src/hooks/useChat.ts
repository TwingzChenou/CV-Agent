
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
        setCurrentTool(null);

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: query, stream: true }),
            });

            if (!response.ok) {
                throw new Error('Failed to fetch response');
            }

            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('Response body is not readable');
            }

            const decoder = new TextDecoder("utf-8");
            let accumulatedText = "";
            let assistantMessageAdded = false;
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                const lines = buffer.split("\n");
                // Maintain the last incomplete line in the buffer
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.trim()) continue;

                    try {
                        const parsed = JSON.parse(line);
                        if (parsed.type === "status") {
                            setCurrentTool(parsed.content);
                            setStatus('loading');
                        } else if (parsed.type === "text") {
                            setStatus('streaming');
                            accumulatedText += parsed.content;
                            
                            setMessages((prev) => {
                                const newMessages = [...prev];
                                if (assistantMessageAdded && newMessages.length > 0) {
                                    const lastMsg = newMessages[newMessages.length - 1];
                                    if (lastMsg.role === 'assistant') {
                                        lastMsg.content = accumulatedText;
                                        return newMessages;
                                    }
                                }
                                
                                // Otherwise append new message
                                newMessages.push({ role: 'assistant', content: accumulatedText });
                                assistantMessageAdded = true;
                                return newMessages;
                            });
                        }
                    } catch (err) {
                        console.error("Error parsing stream chunk:", err, "Line:", line);
                    }
                }
            }

            setStatus('idle');
            setCurrentTool(null);

        } catch (error) {
            console.error("Chat error:", error);
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: "Sorry, something went wrong. Please check if the backend is running." }
            ]);
            setStatus('idle');
            setCurrentTool(null);
        }
    };

    return {
        messages,
        sendMessage,
        status,
        currentTool
    };
}
