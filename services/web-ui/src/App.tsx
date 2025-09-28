import { useState, useEffect } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { Sidebar } from './components/Sidebar';
import { trpc } from './utils/trpc';

function App() {
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0); // Force sidebar refresh

  const createChatMutation = (trpc as any).chats.create.useMutation();

  // Load existing chats on mount
  const { data: chatsData } = (trpc as any).chats.list.useQuery({
    limit: 1,
    sortBy: 'updatedAt',
    sortOrder: 'desc',
  });

  // Select the most recent chat on load if no chat is selected
  useEffect(() => {
    if (!selectedChatId && chatsData?.chats && chatsData.chats.length > 0) {
      setSelectedChatId(chatsData.chats[0].id);
    }
  }, [chatsData, selectedChatId]);

  const handleNewChat = () => {
    if (isCreatingChat) return;

    setIsCreatingChat(true);
    createChatMutation.mutate(
      {
        title: `Chat ${new Date().toLocaleString('hr-HR', {
          day: '2-digit',
          month: '2-digit',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        })}`,
        feature: 'narodne-novine',
        visibility: 'private',
        ragConfig: {
          language: 'hr',
          maxDocuments: 5,
          minConfidence: 0.7,
          temperature: 0.7,
        },
      },
      {
        onSuccess: (data: any) => {
          console.log('Chat created successfully:', data);
          setSelectedChatId(data.id);
          setIsCreatingChat(false);
          // Trigger sidebar refresh
          setRefreshKey(prev => prev + 1);
        },
        onError: (error: any) => {
          console.error('Failed to create chat:', error);
          setIsCreatingChat(false);
          alert(`Failed to create chat: ${error.message || 'Unknown error'}`);
        },
      }
    );
  };

  const handleSelectChat = (chatId: string) => {
    setSelectedChatId(chatId);
  };

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Sidebar */}
      <Sidebar
        key={refreshKey} // Force refresh when key changes
        selectedChatId={selectedChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-gray-800 border-b border-gray-700 px-6 py-3">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-100">
                RAG System - Narodne Novine
              </h1>
              <p className="text-sm text-gray-400 mt-1">
                Croatian Legal Documents Assistant
              </p>
            </div>

            {selectedChatId && (
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span className="flex items-center gap-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                  </svg>
                  Chat ID: {selectedChatId.slice(0, 8)}...
                </span>
              </div>
            )}
          </div>
        </header>

        {/* Chat Area */}
        <main className="flex-1 overflow-hidden">
          {selectedChatId ? (
            <ChatInterface
              key={selectedChatId} // Force remount when chat changes
              chatId={selectedChatId}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <svg className="w-16 h-16 mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <p className="text-lg mb-2">No chat selected</p>
              <p className="text-sm">Create a new chat or select an existing one to start</p>
              <button
                onClick={handleNewChat}
                className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
                disabled={isCreatingChat}
              >
                {isCreatingChat ? (
                  <>
                    <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    Creating...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    New Chat
                  </>
                )}
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;