#!/bin/bash
# Integration test script for Web API → FastAPI → RAG system

echo "=== RAG Integration Test ==="
echo ""

# 1. Test Health endpoints
echo "1. Testing Health Endpoints..."
echo "   - Web API Health:"
curl -s http://localhost:3000/api/v1/health | jq '.'
echo ""
echo "   - FastAPI Health:"
curl -s http://localhost:8081/health | jq '.'
echo ""

# 2. Create a chat
echo "2. Creating a new chat..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:3000/api/trpc/chats.create \
  -H "Content-Type: application/json" \
  -d '{
    "0": {
      "json": {
        "title": "Test RAG Integration Chat",
        "feature": "user",
        "visibility": "private",
        "ragConfig": {
          "language": "hr",
          "maxDocuments": 5,
          "minConfidence": 0.7,
          "temperature": 0.7
        }
      }
    }
  }')

echo "$CHAT_RESPONSE" | jq '.'
CHAT_ID=$(echo "$CHAT_RESPONSE" | jq -r '.[0].result.data.id')
echo "   Chat ID: $CHAT_ID"
echo ""

# 3. Send a message to test RAG
echo "3. Sending a test message to RAG..."
MESSAGE_RESPONSE=$(curl -s -X POST http://localhost:3000/api/trpc/messages.send \
  -H "Content-Type: application/json" \
  -d "{
    \"0\": {
      \"json\": {
        \"chatId\": \"$CHAT_ID\",
        \"content\": \"What is RAG?\",
        \"ragConfig\": {
          \"maxDocuments\": 3,
          \"minConfidence\": 0.6,
          \"temperature\": 0.8
        }
      }
    }
  }")

echo "$MESSAGE_RESPONSE" | jq '.[0].result.data | {
  userMessage: .userMessage.content,
  assistantResponse: .assistantMessage.content,
  documentsUsed: .assistantMessage.metadata.ragContext.documentsUsed,
  confidence: .assistantMessage.metadata.ragContext.confidence,
  responseTime: .assistantMessage.metadata.responseTimeMs
}'
echo ""

# 4. List messages in the chat
echo "4. Listing messages in chat..."
curl -s -X POST http://localhost:3000/api/trpc/messages.list \
  -H "Content-Type: application/json" \
  -d "{
    \"0\": {
      \"json\": {
        \"chatId\": \"$CHAT_ID\",
        \"limit\": 10
      }
    }
  }" | jq '.[0].result.data.messages | map({
    role: .role,
    content: (.content | split("\n")[0:2] | join("\n")),
    timestamp: .createdAt
  })'

echo ""
echo "=== Integration Test Complete ===
"