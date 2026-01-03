import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 p-4">
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">CV Agent IA</h1>
        <p className="text-gray-400">Posez vos questions sur le parcours de Quentin</p>
      </div>

      <ChatInterface />
    </main>
  );
}
