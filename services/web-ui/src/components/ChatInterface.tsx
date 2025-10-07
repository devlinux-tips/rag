import { useState, useRef, useEffect } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { trpc } from '../utils/trpc';

interface ChatInterfaceProps {
  chatId: string;
  onMessageSent?: () => void; // Optional callback when message is sent
}

export function ChatInterface({ chatId, onMessageSent }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<any[]>([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sendMessageMutation = (trpc as any).messages.send.useMutation();

  // Load existing messages for this chat
  const { data: messagesData, isLoading, error } = (trpc as any).messages.list.useQuery(
    {
      chatId,
      limit: 100,
      order: 'asc',
    },
    {
      enabled: !!chatId,
      refetchOnMount: true,
      staleTime: 0, // Always refetch when switching chats
      cacheTime: 0, // Don't cache between chat switches
    }
  );

  // Update messages when data changes
  useEffect(() => {
    if (messagesData?.messages) {
      setMessages(messagesData.messages);
    } else if (messagesData && !messagesData.messages) {
      setMessages([]); // Empty array if no messages in response
    }
    setIsLoadingMessages(false);
  }, [messagesData]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Reset messages when chat changes
  useEffect(() => {
    setMessages([]);
    setIsLoadingMessages(true);
  }, [chatId]);

  const handleSendMessage = async (content: string) => {
    // Add user message immediately
    const userMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      createdAt: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);

    // Send to API
    sendMessageMutation.mutate(
      {
        chatId,
        content,
        ragConfig: {
          language: 'hr',
          maxDocuments: 5,
        },
      },
      {
        onSuccess: (data: any) => {
          // Replace temp user message with real one and add assistant message
          setMessages(prev => [
            ...prev.filter(m => m.id !== userMessage.id),
            data.userMessage,
            data.assistantMessage,
          ]);
          // Notify parent that a message was sent (to update sidebar)
          onMessageSent?.();
        },
        onError: (error: any) => {
          // AI-FRIENDLY ERROR LOG: Structured for quick AI pattern recognition
          console.error('ERROR_SEND_MESSAGE | CONTEXT:', {
            error_type: 'MESSAGE_SEND_FAILED',
            error_code: error?.code || error?.data?.code || 'UNKNOWN',
            error_message: error?.message,
            error_data: error?.data,
            error_cause: error?.cause,
            error_stack: error?.stack,
            context_chat_id: chatId,
            context_timestamp: new Date().toISOString(),
            context_url: window.location.href,
            http_status: error?.response?.status,
            trpc_path: error?.shape?.data?.path,
          });

          // Add user-friendly error message (technical details logged only)
          setMessages(prev => [
            ...prev,
            {
              id: `error-${Date.now()}`,
              role: 'assistant',
              content: 'Žao mi je, dogodila se greška pri obradi pitanja. Molim pokušajte ponovno.',
              createdAt: new Date().toISOString(),
              isError: true,
            },
          ]);
        },
      }
    );
  };

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      <div className="flex-1 overflow-y-auto">
        {isLoading && isLoadingMessages ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-400">Loading messages...</div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-red-400">Failed to load messages</div>
          </div>
        ) : (
          <>
            <MessageList
              messages={messages}
              isLoading={sendMessageMutation.isLoading}
            />
            <div ref={messagesEndRef} />
          </>
        )}
      </div>
      
      <MessageInput 
        onSendMessage={handleSendMessage}
        disabled={sendMessageMutation.isLoading}
      />
    </div>
  );
}