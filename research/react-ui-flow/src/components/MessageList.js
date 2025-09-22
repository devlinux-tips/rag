import React, { useEffect, useRef } from 'react';
import Message from './Message';

const MessageList = ({ messages, isLoading, error }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  return (
    <div className="message-list">
      <div className="message-container">
        {messages.length === 0 && !isLoading && (
          <div className="message bot">
            <div className="message-header">RAG Assistant</div>
            <div className="message-content">
              <p>
                Hello! I'm your RAG Assistant. Ask me anything and I'll help you find information from the knowledge base.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, index) => (
          <Message
            key={index}
            message={msg.content}
            isUser={msg.isUser}
          />
        ))}

        {isLoading && (
          <div className="message bot">
            <div className="message-header">RAG Assistant</div>
            <div className="message-content">
              <div className="loading-indicator">
                <span>Thinking</span>
                <div className="loading-dots">
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                  <div className="loading-dot"></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default MessageList;