#!/bin/bash
echo "=== Testing Narodne Novine Query with OpenRouter ==="
echo ""

# 1. Create a new Narodne Novine chat
echo "1. Creating new Narodne Novine chat..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:3000/api/trpc/chats.create \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Environmental Regulations Query",
    "feature": "narodne-novine",
    "visibility": "private",
    "ragConfig": {
      "language": "hr",
      "maxDocuments": 5,
      "minConfidence": 0.7,
      "temperature": 0.7
    }
  }')

CHAT_ID=$(echo "$CHAT_RESPONSE" | jq -r '.result.data.id')
echo "   Chat ID: $CHAT_ID"
echo ""

# 2. Send the query about environmental protection and fuels
echo "2. Sending query: 'Što govore propisi o zaštiti okoliša i gorivima?'"
MESSAGE_RESPONSE=$(curl -s -X POST http://localhost:3000/api/trpc/messages.send \
  -H "Content-Type: application/json" \
  -d "{
    \"chatId\": \"$CHAT_ID\",
    \"content\": \"Što govore propisi o zaštiti okoliša i gorivima?\"
  }")

# 3. Extract and display the response
echo ""
echo "3. Response from OpenRouter/Qwen3:"
echo "=================================="
echo "$MESSAGE_RESPONSE" | jq -r '.assistantMessage.content // .error.message'
echo "=================================="
echo ""

# 4. Show metadata
echo "4. Response Metadata:"
echo "$MESSAGE_RESPONSE" | jq '.assistantMessage.metadata // {}'

