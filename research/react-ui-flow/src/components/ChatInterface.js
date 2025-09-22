import React, { useState } from 'react';
import MessageList from './MessageList';
import InputArea from './InputArea';
import { v4 as uuidv4 } from 'uuid';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatId] = useState(() => uuidv4());

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

  const sendMessage = async (messageText) => {
    // Clear any previous errors
    setError(null);

    // Add user message to the conversation
    const userMessage = {
      content: messageText,
      isUser: true,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Prepare the request payload to match the server's SendMessageRequest model
      const requestPayload = {
        conversation_id: chatId,
        message: messageText,
        language: 'hr',
        tenant_slug: 'development',
        user_id: 'dev_user',
        max_rag_results: 3
      };

      // Make API call to the RAG service
      const response = await fetch(`${API_BASE_URL}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Extract content from the response
      let botContent = '';
      if (data.content) {
        botContent = data.content;
      } else if (data.response && data.response.content) {
        botContent = data.response.content;
      } else if (typeof data === 'string') {
        botContent = data;
      } else {
        botContent = 'I received your message but couldn\'t generate a proper response.';
      }

      // Add bot response to the conversation
      const botMessage = {
        content: botContent,
        isUser: false,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError(`Failed to send message: ${err.message}`);

      // Optionally add a generic error response
      const errorMessage = {
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        isUser: false,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>RAG Chat Assistant</h1>
      </div>

      <MessageList
        messages={messages}
        isLoading={isLoading}
        error={error}
      />

      <InputArea
        onSendMessage={sendMessage}
        isLoading={isLoading}
      />
    </div>
  );
};

export default ChatInterface;