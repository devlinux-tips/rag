import { useState } from 'react';
import clsx from 'clsx';
import type { Source } from '../types/message';

interface SourcesListProps {
  sources: Source[];
}

export function SourcesList({ sources }: SourcesListProps) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="sources-section mt-4 pt-3 border-t border-gray-700">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left text-sm text-gray-300 hover:text-gray-100 transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <span>📄</span>
          <span className="font-medium">Izvori</span>
          <span className="text-gray-500">({sources.length} {sources.length === 1 ? 'dokument' : sources.length < 5 ? 'dokumenta' : 'dokumenata'})</span>
        </span>
        <svg
          className={clsx(
            'w-4 h-4 transition-transform ml-auto',
            expanded ? 'rotate-180' : ''
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="sources-list mt-3 space-y-2 animate-fade-in">
          {sources.map((source) => (
            <div
              key={source.citationId}
              className="source-card p-3 bg-gray-900/50 border border-gray-700 rounded-lg hover:border-gray-600 transition-colors"
            >
              <div className="source-header flex items-center gap-2 mb-2">
                <span className="citation-number inline-flex items-center justify-center w-6 h-6 bg-blue-600 text-white text-xs font-bold rounded">
                  {source.citationId}
                </span>
                <span className="source-issue text-sm font-medium text-blue-400">
                  {source.issue}
                </span>
              </div>

              <h4 className="source-title text-sm font-medium text-gray-200 mb-2 leading-snug">
                {source.title}
              </h4>

              <div className="source-meta flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-400 mb-2">
                {source.publisher && (
                  <span className="flex items-center gap-1">
                    <span>📋</span>
                    <span>{source.publisher}</span>
                  </span>
                )}
                {source.year && (
                  <span className="flex items-center gap-1">
                    <span>📅</span>
                    <span>{source.year}</span>
                  </span>
                )}
                {source.relevance !== undefined && (
                  <span className="flex items-center gap-1">
                    <span>⭐</span>
                    <span>Relevantnost: {Math.round(source.relevance)}%</span>
                  </span>
                )}
              </div>

              <a
                href={source.eli}
                target="_blank"
                rel="noopener noreferrer"
                className="source-link inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                <span>🔗</span>
                <span>Pogledaj na Narodnim novinama</span>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
