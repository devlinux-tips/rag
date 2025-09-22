# React Chat UI for RAG System

A modern React-based chat interface for the RAG (Retrieval-Augmented Generation) system with full markdown rendering support.

## Features

- Dark theme ChatGPT-style interface
- Real-time chat messaging
- Full markdown rendering for bot responses including:
  - Headers, lists, bold, italic text
  - Code blocks with syntax highlighting
  - Tables, blockquotes, links
  - GitHub Flavored Markdown (GFM) support
- Responsive design
- Auto-scrolling message list
- Loading states and error handling
- Clean, modern UI with proper spacing

## Setup Instructions

1. **Install dependencies:**
   ```bash
   cd /home/x/src/rag/learn-rag/research/react-ui-flow
   npm install
   ```

2. **Configure API endpoint:**
   ```bash
   cp .env.example .env
   # Edit .env to set your API endpoint
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000` to use the chat interface.

## API Integration

The chat interface sends requests to your RAG backend with the following format:

**Request:**
```json
{
  "chat_id": "uuid-string",
  "message": "user input text",
  "tenant_slug": "development",
  "user_id": "dev_user"
}
```

**Expected Response:**
The backend should return a JSON response containing a `content` field with markdown text that will be properly rendered in the chat interface.

## Project Structure

```
src/
├── App.js                 # Main application component
├── index.js              # React entry point
├── components/
│   ├── ChatInterface.js  # Main chat container with state management
│   ├── MessageList.js    # Message display area with auto-scroll
│   ├── Message.js        # Individual message component with markdown
│   └── InputArea.js      # Text input and send button
└── styles/
    └── Chat.css          # Dark theme styling
```

## Customization

- **API Endpoint:** Set `REACT_APP_API_URL` in your `.env` file
- **Styling:** Modify `/src/styles/Chat.css` for appearance changes
- **Default Values:** Update tenant_slug and user_id in `ChatInterface.js`

## Dependencies

- **react-markdown:** Full markdown rendering with GFM support
- **react-syntax-highlighter:** Code block syntax highlighting
- **uuid:** Unique chat session identifiers