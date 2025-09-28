import { Message } from './Message';
import { TypingIndicator } from './TypingIndicator';

interface MessageListProps {
  messages: any[];
  isLoading: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 px-4">
        <div className="text-center">
          <p className="text-lg mb-2">Welcome to RAG Chat</p>
          <p className="text-sm">Ask me anything about Croatian legal documents</p>
        </div>
      </div>
    );
  }

  return (
    <div className="px-4 py-8">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      {isLoading && <TypingIndicator />}
    </div>
  );
}