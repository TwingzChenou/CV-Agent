import Sidebar from '@/components/Sidebar';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-950">
      <div className="flex flex-col lg:flex-row">

        {/* 1. La Sidebar (Fixe sur Desktop) */}
        {/* Elle prend 40% de la largeur sur grand écran */}
        <Sidebar />

        {/* 2. La Zone de Chat */}
        {/* On ajoute une marge à gauche (lg:ml-[40%]) pour ne pas qu'elle passe SOUS la sidebar fixe */}
        <main className="w-full lg:w-[60%] lg:ml-[40%] min-h-screen relative">
          <ChatInterface />
        </main>

      </div>
    </div>
  );
}