import React from 'react';
import { Linkedin, Github, Mail, MapPin, Briefcase, FileText } from 'lucide-react';

const Sidebar = () => {
    return (
        <aside className="w-full lg:w-[40%] lg:h-screen bg-zinc-950 lg:fixed lg:left-0 lg:top-0 p-8 lg:p-12 flex flex-col justify-center items-center animate-slide-in-left">
            {/* Radial gradient background */}
            <div className="absolute inset-0 bg-gradient-radial from-blue-950/30 via-zinc-950 to-zinc-950 pointer-events-none" />

            <div className="relative z-10 w-full max-w-md">
                {/* Profile Photo */}
                <div className="flex justify-center mb-8">
                    <div className="relative group">
                        <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-full blur opacity-75 group-hover:opacity-100 transition duration-300 animate-pulse-glow" />
                        <div className="relative w-40 h-40 rounded-full overflow-hidden border-4 border-zinc-900">
                            <div className="w-full h-full bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center text-white text-6xl font-bold">
                                QF
                            </div>
                        </div>
                    </div>
                </div>

                {/* Name & Title */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl lg:text-5xl font-bold text-white mb-3 tracking-tight">
                        Quentin Forget
                    </h1>
                    <p className="text-xl lg:text-2xl gradient-text font-semibold mb-4">
                        Data Scientist & AI Engineer
                    </p>
                    <div className="flex items-center justify-center gap-2 text-zinc-400 text-sm">
                        <MapPin size={16} />
                        <span>Paris, France</span>
                    </div>
                </div>

                {/* Bio */}
                <div className="p-6 mb-8">
                    <p className="text-zinc-300 text-center leading-relaxed">
                        Je transforme vos solutions traditionnelles en moteurs d'intelligence artificielle performants, alliant optimisation des coûts et innovation technologique.
                    </p>
                </div>

                {/* Skills Tags */}
                <div className=" p-6 mb-8">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                        <Briefcase size={18} className="text-blue-400" />
                        Compétences clés
                    </h3>
                    <div className="flex flex-wrap gap-2">
                        {['Python', 'Scikit-learn', 'TensorFlow', 'LlamaIndex', 'Pinecone', 'Docker', 'Git', 'FastAPI', 'MongoDB'].map((skill) => (
                            <span
                                key={skill}
                                className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-full text-xs text-zinc-300 hover:bg-white/10 hover:border-blue-500/50 transition-all duration-300 cursor-default"
                            >
                                {skill}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Social Links */}
                <div className="flex justify-center gap-4">
                    <a
                        href="https://www.linkedin.com/in/quentin-forget-197705230"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="glass-strong w-12 h-12 rounded-full flex items-center justify-center text-zinc-400 hover:text-blue-400 hover:border-blue-500/50 transition-all duration-300 group"
                        aria-label="LinkedIn"
                    >
                        <Linkedin size={20} className="group-hover:scale-110 transition-transform" />
                    </a>
                    <a
                        href="https://github.com/TwingzChenou"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="glass-strong w-12 h-12 rounded-full flex items-center justify-center text-zinc-400 hover:text-emerald-400 hover:border-emerald-500/50 transition-all duration-300 group"
                        aria-label="GitHub"
                    >
                        <Github size={20} className="group-hover:scale-110 transition-transform" />
                    </a>
                    <a
                        href="/CV_Quentin_Forget.pdf"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="glass-strong w-12 h-12 rounded-full flex items-center justify-center text-zinc-400 hover:text-red-400 hover:border-red-500/50 transition-all duration-300 group"
                        aria-label="CV PDF"
                    >
                        <FileText size={20} className="group-hover:scale-110 transition-transform" />
                    </a>
                    <a
                        href="mailto:quentin-forget@hotmail.fr"
                        className="glass-strong w-12 h-12 rounded-full flex items-center justify-center text-zinc-400 hover:text-purple-400 hover:border-purple-500/50 transition-all duration-300 group"
                        aria-label="Email"
                    >
                        <Mail size={20} className="group-hover:scale-110 transition-transform" />
                    </a>
                </div>

                {/* Decorative element */}
                <div className="mt-8 flex justify-center">
                    <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-full" />
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;