import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const Message = ({ message, isUser }) => {
  const messageClass = isUser ? 'message user' : 'message bot';
  const senderName = isUser ? 'You' : 'RAG Assistant';

  return (
    <div className={messageClass}>
      <div className="message-header">{senderName}</div>
      <div className="message-content">
        {isUser ? (
          // User messages are plain text
          <p>{message}</p>
        ) : (
          // Bot messages are rendered as markdown
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{
                      background: '#1a1a1a',
                      border: '1px solid #3a3a3a',
                      borderRadius: '8px',
                    }}
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
};

export default Message;