import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import clsx from 'clsx';

interface MessageProps {
  message: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    createdAt: string;
    isError?: boolean;
    metadata?: {
      ragContext?: {
        documentsRetrieved?: number;
        documentsUsed?: number;
        searchTimeMs?: number;
        responseTimeMs?: number;
        tokensUsed?: {
          input?: number;
          output?: number;
          total?: number;
        };
      };
    };
  };
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className={clsx(
      'flex gap-4 mb-6 message-fade-in',
      isUser ? 'justify-end' : 'justify-start'
    )}>
      <div className={clsx(
        'flex gap-3 max-w-[85%]',
        isUser && 'flex-row-reverse'
      )}>
        {/* Avatar */}
        <div className={clsx(
          'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
          isUser ? 'bg-blue-600' : 'bg-green-600'
        )}>
          <span className="text-white text-sm font-semibold">
            {isUser ? 'U' : 'AI'}
          </span>
        </div>

        {/* Message content */}
        <div className={clsx(
          'px-5 py-4 rounded-lg',
          isUser
            ? 'bg-blue-600 text-white'
            : message.isError
            ? 'bg-red-900/50 text-red-200 border border-red-700'
            : 'bg-gray-800 text-gray-100'
        )}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              className="prose prose-invert max-w-none prose-p:my-3 prose-ul:my-3 prose-ol:my-3 prose-li:my-1 prose-headings:mt-4 prose-headings:mb-2 prose-pre:my-3"
              components={{
                code({ className, children, ...props }: any) {
                  const inline = props.inline || false;
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark as any}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code
                      className="bg-gray-700 px-1 py-0.5 rounded text-sm"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                },
                // Style tables
                table: ({ children }) => (
                  <table className="border-collapse border border-gray-600 my-4">
                    {children}
                  </table>
                ),
                th: ({ children }) => (
                  <th className="border border-gray-600 px-3 py-2 bg-gray-800">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="border border-gray-600 px-3 py-2">
                    {children}
                  </td>
                ),
                // Style lists with better spacing
                ul: ({ children }) => (
                  <ul className="list-disc list-inside my-3 space-y-1.5 pl-6">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside my-3 space-y-1.5 pl-6">
                    {children}
                  </ol>
                ),
                li: ({ children }) => (
                  <li className="my-1 leading-relaxed">
                    {children}
                  </li>
                ),
                // Style paragraphs with more breathing room
                p: ({ children }) => (
                  <p className="my-3 leading-relaxed">
                    {children}
                  </p>
                ),
                // Style links
                a: ({ children, href }) => (
                  <a
                    href={href}
                    className="text-blue-400 hover:text-blue-300 underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {children}
                  </a>
                ),
                // Style blockquotes
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-gray-600 pl-4 italic my-3">
                    {children}
                  </blockquote>
                ),
                // Style headings with more space
                h1: ({ children }) => (
                  <h1 className="text-2xl font-bold mt-4 mb-2">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-xl font-bold mt-3.5 mb-2">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-lg font-semibold mt-3 mb-1.5">{children}</h3>
                ),
                // Style strong/bold text
                strong: ({ children }) => (
                  <strong className="font-semibold text-gray-50">{children}</strong>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}

          {/* Copy button and metadata for assistant messages */}
          {!isUser && !message.isError && (
            <div className="mt-4 pt-3 border-t border-gray-700">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-xs text-gray-400">
                  {message.metadata?.ragContext && (
                    <>
                      {message.metadata.ragContext.documentsRetrieved && (
                        <span>📄 {message.metadata.ragContext.documentsUsed || 0}/{message.metadata.ragContext.documentsRetrieved} docs</span>
                      )}
                      {message.metadata.ragContext.searchTimeMs && (
                        <span>🔍 {(message.metadata.ragContext.searchTimeMs / 1000).toFixed(1)}s</span>
                      )}
                      {message.metadata.ragContext.tokensUsed?.total && (
                        <span>🎯 {message.metadata.ragContext.tokensUsed.total} tokens</span>
                      )}
                      {!message.metadata.ragContext.tokensUsed?.total && message.metadata.ragContext.responseTimeMs && (
                        <span>⏱️ {(message.metadata.ragContext.responseTimeMs / 1000).toFixed(1)}s</span>
                      )}
                    </>
                  )}
                </div>

                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded transition-colors"
                >
                  {copied ? (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}