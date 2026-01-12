'use client';

import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Send, Loader2, User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { useChat } from '@/hooks/useChat';

const ChatWidget = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [input, setInput] = useState('');

    // 👇 On utilise le hook au lieu de réécrire le fetch
    const { messages, sendMessage, status } = useChat();

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isOpen]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || status === 'loading') return;

        const query = input;
        setInput(''); // On vide l'input tout de suite
        await sendMessage(query); // On laisse le hook gérer l'envoi
    };

    return (
        <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end sm:bottom-6 sm:right-6 font-sans">

            {/* Fenêtre de chat */}
            {isOpen && (
                <div className="mb-4 w-[350px] sm:w-[400px] h-[500px] bg-zinc-900 rounded-xl shadow-2xl flex flex-col border border-zinc-800 overflow-hidden animate-in slide-in-from-bottom-10 fade-in duration-300">

                    {/* Header */}
                    <div className="flex items-center justify-between p-4 bg-zinc-950 border-b border-zinc-800">
                        <div className="flex items-center gap-3">
                            <div className="relative">
                                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse absolute top-0 right-0" />
                                <Bot size={20} className="text-zinc-100" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-zinc-100 text-sm">Quentin AI</h3>
                                <p className="text-xs text-zinc-500">Répond instantanément</p>
                            </div>
                        </div>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="text-zinc-500 hover:text-white transition-colors p-1 hover:bg-zinc-800 rounded-full"
                        >
                            <X size={18} />
                        </button>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-zinc-900/95 scrollbar-thin scrollbar-thumb-zinc-700">
                        {/* Message de bienvenue par défaut si la liste est vide */}
                        {messages.length === 0 && (
                            <div className="text-center text-zinc-500 text-sm mt-10 px-6">
                                👋 Bonjour ! Je suis l'assistant virtuel de Quentin. Posez-moi une question sur son CV.
                            </div>
                        )}

                        {messages.map((msg, index) => (
                            <div key={index} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border border-white/5 ${msg.role === 'user' ? 'bg-blue-600/20 text-blue-400' : 'bg-emerald-600/20 text-emerald-400'}`}>
                                    {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                                </div>

                                <div className={`max-w-[85%] rounded-2xl p-3 text-sm leading-relaxed ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white rounded-tr-sm'
                                    : 'bg-zinc-800 text-zinc-200 border border-zinc-700 rounded-tl-sm'
                                    }`}>
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                </div>
                            </div>
                        ))}

                        {status === 'loading' && (
                            <div className="flex gap-2 items-center text-zinc-500 text-xs ml-2">
                                <Loader2 size={12} className="animate-spin" />
                                <span>Quentin écrit...</span>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSubmit} className="p-3 bg-zinc-950 border-t border-zinc-800">
                        <div className="relative flex items-center">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Posez une question..."
                                className="w-full pl-4 pr-10 py-3 bg-zinc-900 border border-zinc-800 rounded-xl text-zinc-100 placeholder-zinc-500 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 focus:outline-none text-sm transition-all"
                                disabled={status === 'loading'}
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || status === 'loading'}
                                className="absolute right-2 p-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:bg-zinc-700 rounded-lg text-white transition-all"
                            >
                                <Send size={16} />
                            </button>
                        </div>
                    </form>
                </div>
            )}

            {/* Bouton Flottant (Toggle) */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`p-4 rounded-full shadow-2xl transition-all duration-300 transform hover:scale-105 active:scale-95 flex items-center justify-center ${isOpen ? 'bg-zinc-800 text-zinc-400 rotate-90' : 'bg-gradient-to-r from-blue-600 to-emerald-600 text-white'
                    }`}
            >
                {isOpen ? <X size={24} /> : <MessageCircle size={28} />}
            </button>
        </div>
    );
};

export default ChatWidget;