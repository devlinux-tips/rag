export function TypingIndicator() {
  return (
    <div className="flex gap-4 mb-6">
      <div className="flex gap-3">
        <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center flex-shrink-0">
          <span className="text-white text-sm font-semibold">AI</span>
        </div>
        <div className="px-4 py-3 rounded-lg bg-gray-800">
          <div className="flex gap-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
          </div>
        </div>
      </div>
    </div>
  );
}