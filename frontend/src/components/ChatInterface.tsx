
"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Sparkles, User, Bot, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { useChat, Message } from "@/hooks/useChat";
import { cn } from "@/lib/utils";

export default function ChatInterface() {
    const { messages, sendMessage, status, currentTool } = useChat();
    const [inputValue, setInputValue] = useState("");
    const [hasStarted, setHasStarted] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages, status]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!inputValue.trim()) return;

        if (!hasStarted) setHasStarted(true);

        const query = inputValue;
        setInputValue("");
        await sendMessage(query);
    };

    return (
        <div className="flex flex-col h-screen bg-zinc-950 text-white overflow-hidden relative font-sans">
            {/* Background Gradients */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-[100px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-500/10 rounded-full blur-[100px]" />
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col relative z-10 w-full max-w-4xl mx-auto p-4 md:p-6">

                {/* Messages List (Visible only after start) */}
                <AnimatePresence>
                    {hasStarted && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.2 }}
                            className="flex-1 overflow-y-auto space-y-6 pb-24 scrollbar-hide"
                        >
                            {messages.map((msg, idx) => (
                                <MessageBubble key={idx} message={msg} />
                            ))}

                            {/* Live Feedback / Tool Call Indicator */}
                            {status === 'loading' && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex items-center gap-2 text-zinc-400 text-sm ml-2"
                                >
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Thinking...</span>
                                </motion.div>
                            )}
                            <div ref={scrollRef} />
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Hero / Input Area Transition */}
                <motion.div
                    layout
                    initial={{ justifySelf: "center", alignSelf: "center", flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}
                    animate={hasStarted ? {
                        flex: 0,
                        marginTop: "auto",
                        justifyContent: "flex-end"
                    } : {
                        flex: 1,
                        justifyContent: "center"
                    }}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.8 }}
                    className={cn(
                        "w-full transition-all relative",
                        !hasStarted ? "flex flex-col items-center justify-center gap-6" : "pb-4"
                    )}
                >
                    {!hasStarted && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="text-center space-y-2 mb-8"
                        >
                            <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
                                Hello, I'm Context.
                            </h1>
                            <p className="text-zinc-400 text-lg md:text-xl">
                                Ask me anything about Quentin's work or experience.
                            </p>
                        </motion.div>
                    )}

                    <form onSubmit={handleSubmit} className="w-full relative max-w-2xl mx-auto">
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/50 to-emerald-500/50 rounded-2xl blur opacity-30 group-hover:opacity-75 transition duration-1000 group-hover:duration-200" />
                            <div className="relative bg-zinc-900/90 backdrop-blur-xl rounded-2xl border border-white/10 flex items-center p-2 shadow-2xl">
                                <Sparkles className="w-5 h-5 text-blue-400 ml-3 mr-3" />
                                <input
                                    type="text"
                                    value={inputValue}
                                    onChange={(e) => setInputValue(e.target.value)}
                                    placeholder="Start typing..."
                                    className="flex-1 bg-transparent border-none outline-none text-white placeholder-zinc-500 h-10"
                                    autoFocus
                                    suppressHydrationWarning={true}
                                />
                                <button
                                    type="submit"
                                    disabled={!inputValue.trim() || status === 'loading'}
                                    className="p-2 rounded-xl bg-white/10 hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-white"
                                >
                                    <Send className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    </form>
                </motion.div>

            </div>
        </div>
    );
}

function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === 'user';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            className={cn(
                "flex w-full",
                isUser ? "justify-end" : "justify-start"
            )}
        >
            <div className={cn(
                "flex gap-3 max-w-[85%] md:max-w-[75%]",
                isUser ? "flex-row-reverse" : "flex-row"
            )}>
                {/* Avatar */}
                <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border border-white/10",
                    isUser ? "bg-blue-600/20 text-blue-400" : "bg-emerald-600/20 text-emerald-400"
                )}>
                    {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>

                {/* Content */}
                <div className={cn(
                    "p-4 rounded-2xl text-sm md:text-base leading-relaxed backdrop-blur-md border shadow-sm",
                    isUser
                        ? "bg-blue-600/10 border-blue-500/20 text-blue-50 rounded-tr-sm"
                        : "bg-zinc-800/50 border-white/5 text-zinc-100 rounded-tl-sm"
                )}>
                    <ReactMarkdown
                        components={{
                            code: ({ node, className, children, ...props }) => { // Removed redundant 'inline'
                                return (
                                    <code className={cn("bg-zinc-900/50 px-1 py-0.5 rounded text-xs font-mono text-emerald-300", className)} {...props}>
                                        {children}
                                    </code>
                                );
                            }
                        }}
                    >
                        {message.content}
                    </ReactMarkdown>
                </div>
            </div>
        </motion.div>
    )
}
