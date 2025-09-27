#!/bin/bash
# Test script for Narodne Novine RAG integration

echo "=== Narodne Novine RAG Integration Test ==="
echo ""

# 1. Create a Narodne Novine chat
echo "1. Creating Narodne Novine chat..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:3000/api/trpc/chats.create \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Narodne Novine Query Session",
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
echo "   Feature: narodne-novine"
echo "   Collection: Features_narodne-novine_hr"
echo ""

# 2. Test queries about Narodne Novine
echo "2. Sending test queries..."
echo ""

# Query 1: What are Narodne Novine?
echo "   Query 1: 'Što su Narodne novine?'"
MESSAGE1=$(curl -s -X POST http://localhost:3000/api/trpc/messages.send \
  -H "Content-Type: application/json" \
  -d "{
    \"chatId\": \"$CHAT_ID\",
    \"content\": \"Što su Narodne novine?\"
  }")
echo "   Response received (truncated)"
echo ""

# Query 2: Legal document question
echo "   Query 2: 'Koji zakoni su objavljeni u zadnjih mjesec dana?'"
MESSAGE2=$(curl -s -X POST http://localhost:3000/api/trpc/messages.send \
  -H "Content-Type: application/json" \
  -d "{
    \"chatId\": \"$CHAT_ID\",
    \"content\": \"Koji zakoni su objavljeni u zadnjih mjesec dana?\"
  }")
echo "   Response received (truncated)"
echo ""

# 3. Show chat summary
echo "3. Chat Summary:"
curl -s -X POST http://localhost:3000/api/trpc/chats.getById \
  -H "Content-Type: application/json" \
  -d "{
    \"chatId\": \"$CHAT_ID\"
  }" | jq '{
    id: .result.data.id,
    title: .result.data.title,
    feature: .result.data.feature,
    language: .result.data.language,
    messageCount: .result.data.messageCount,
    createdAt: .result.data.createdAt
  }'

echo ""
echo "=== Test Complete ==="
echo "Collection being queried: Features_narodne-novine_hr"
echo "Note: Responses depend on documents indexed in the RAG system"