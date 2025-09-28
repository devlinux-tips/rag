import { useState } from 'react';
import clsx from 'clsx';
import { trpc } from '../utils/trpc';

interface SidebarProps {
  selectedChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
}

export function Sidebar({ selectedChatId, onSelectChat, onNewChat }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Fetch chat list
  const { data: chatsData, isLoading, error, refetch } = (trpc as any).chats.list.useQuery(
    {
      limit: 50,
      sortBy: 'updatedAt',
      sortOrder: 'desc',
    },
    {
      onError: (err: any) => {
        console.error('Failed to fetch chats:', err);
      },
    }
  );

  const deleteChatMutation = (trpc as any).chats.delete.useMutation({
    onSuccess: () => {
      refetch();
      if (selectedChatId === deletingId) {
        onNewChat();
      }
      setDeletingId(null);
    },
  });

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const updateChatMutation = (trpc as any).chats.update.useMutation({
    onSuccess: () => {
      refetch();
      setEditingId(null);
    },
  });

  const handleEditStart = (chat: any) => {
    setEditingId(chat.id);
    setEditTitle(chat.title);
  };

  const handleEditSave = (chatId: string) => {
    if (editTitle.trim()) {
      updateChatMutation.mutate({
        id: chatId,
        title: editTitle.trim(),
      });
    }
  };

  const handleEditCancel = () => {
    setEditingId(null);
    setEditTitle('');
  };

  const handleDelete = (chatId: string) => {
    if (window.confirm('Are you sure you want to delete this chat?')) {
      setDeletingId(chatId);
      deleteChatMutation.mutate({ id: chatId });
    }
  };

  const formatDate = (date: string) => {
    const d = new Date(date);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString();
  };

  return (
    <div
      className={clsx(
        'bg-gray-900 border-r border-gray-700 flex flex-col transition-all duration-300',
        isCollapsed ? 'w-16' : 'w-80'
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className={clsx(
          'flex items-center',
          isCollapsed ? 'flex-col gap-2' : 'justify-between'
        )}>
          <button
            onClick={onNewChat}
            className={clsx(
              'flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors',
              isCollapsed ? 'px-2 justify-center' : 'flex-1'
            )}
            title="New Chat"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            {!isCollapsed && <span>New Chat</span>}
          </button>

          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className={clsx(
              'p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors',
              isCollapsed ? '' : 'ml-2'
            )}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isCollapsed ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto">
        {error ? (
          <div className="p-4 text-center text-red-400">
            <p>Failed to load chats</p>
            <p className="text-xs mt-1">{error.message}</p>
            <button
              onClick={() => refetch()}
              className="mt-2 text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded"
            >
              Retry
            </button>
          </div>
        ) : isLoading ? (
          <div className="p-4 text-center text-gray-500">
            Loading chats...
          </div>
        ) : !chatsData?.chats || chatsData.chats.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            {!isCollapsed && 'No chats yet. Start a new conversation!'}
          </div>
        ) : (
          <div className="py-2">
            {chatsData.chats.map((chat: any) => (
              <div
                key={chat.id}
                className={clsx(
                  'group relative mx-2 mb-1 rounded-lg transition-all cursor-pointer',
                  selectedChatId === chat.id
                    ? 'bg-gray-800 text-white'
                    : 'hover:bg-gray-800/50 text-gray-300'
                )}
              >
                {editingId === chat.id ? (
                  <div className="flex items-center gap-1 p-2">
                    <input
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleEditSave(chat.id);
                        if (e.key === 'Escape') handleEditCancel();
                      }}
                      className="flex-1 px-2 py-1 bg-gray-700 text-white rounded text-sm"
                      autoFocus
                    />
                    <button
                      onClick={() => handleEditSave(chat.id)}
                      className="p-1 text-green-400 hover:text-green-300"
                      title="Save"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </button>
                    <button
                      onClick={handleEditCancel}
                      className="p-1 text-red-400 hover:text-red-300"
                      title="Cancel"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => onSelectChat(chat.id)}
                    className="flex items-center p-2"
                  >
                    {!isCollapsed && (
                      <>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm truncate">
                            {chat.title}
                          </div>
                          <div className="text-xs text-gray-500 mt-1 flex items-center gap-2">
                            <span>{formatDate(chat.updatedAt)}</span>
                            {chat._count?.messages > 0 && (
                              <span>• {chat._count.messages} messages</span>
                            )}
                          </div>
                        </div>

                        {/* Action buttons - show on hover */}
                        <div className="flex items-center gap-1 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditStart(chat);
                            }}
                            className="p-1 text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded"
                            title="Edit title"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(chat.id);
                            }}
                            className="p-1 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded"
                            title="Delete chat"
                            disabled={deletingId === chat.id}
                          >
                            {deletingId === chat.id ? (
                              <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                  d="M4 12a8 8 0 018-8v8H4z" />
                              </svg>
                            ) : (
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            )}
                          </button>
                        </div>
                      </>
                    )}

                    {isCollapsed && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full mx-auto" />
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer with user info */}
      {!isCollapsed && (
        <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center">
              <span className="text-gray-300 text-sm font-medium">U</span>
            </div>
            <div>
              <div className="text-gray-300">User</div>
              <div className="text-gray-500">Narodne Novine</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}