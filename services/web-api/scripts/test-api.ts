#!/usr/bin/env tsx

/**
 * Test script for the Web API with mock JWT authentication
 * Run with: npm run test:api
 */

import { mockTokens } from '../src/utils/mock-jwt';

const BASE_URL = 'http://localhost:3000';

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
};

function log(message: string, color: string = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

async function testEndpoint(
  name: string,
  method: string,
  path: string,
  body?: any,
  token?: string
) {
  log(`\n📌 Testing: ${name}`, colors.blue);

  try {
    const headers: any = {
      'Content-Type': 'application/json',
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    const data = await response.json();

    if (response.ok) {
      log(`✅ Success (${response.status})`, colors.green);
      console.log(JSON.stringify(data, null, 2));
    } else {
      log(`❌ Failed (${response.status})`, colors.red);
      console.log(JSON.stringify(data, null, 2));
    }

    return data;
  } catch (error) {
    log(`❌ Error: ${error}`, colors.red);
    return null;
  }
}

async function testTRPC(
  procedure: string,
  input: any,
  token?: string,
  type: 'query' | 'mutation' = 'mutation'
) {
  log(`\n📌 Testing tRPC: ${procedure}`, colors.magenta);

  try {
    const headers: any = {
      'Content-Type': 'application/json',
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    // For queries, use GET with query params, for mutations use POST
    const url = type === 'query'
      ? `${BASE_URL}/api/trpc/${procedure}?input=${encodeURIComponent(JSON.stringify(input))}`
      : `${BASE_URL}/api/trpc/${procedure}`;

    const response = await fetch(url, {
      method: type === 'query' ? 'GET' : 'POST',
      headers,
      body: type === 'mutation' ? JSON.stringify(input) : undefined,
    });

    const data = await response.json();

    if (response.ok) {
      log(`✅ Success`, colors.green);
      console.log(JSON.stringify(data.result.data, null, 2));
    } else {
      log(`❌ Failed`, colors.red);
      console.log(JSON.stringify(data, null, 2));
    }

    return data;
  } catch (error) {
    log(`❌ Error: ${error}`, colors.red);
    return null;
  }
}

async function runTests() {
  log('\n╔════════════════════════════════════════╗', colors.yellow);
  log('║       Web API Test Suite               ║', colors.yellow);
  log('╚════════════════════════════════════════╝', colors.yellow);

  // Test 1: Health check (no auth required)
  await testEndpoint('Health Check', 'GET', '/api/v1/health');

  // Test 2: API Info (no auth required)
  await testEndpoint('API Info', 'GET', '/api/v1/info');

  // Test 3: User info without token (should fail)
  await testEndpoint('User Info (No Token)', 'GET', '/api/v1/user');

  // Test 4: User info with mock auth (AUTH_MODE=mock)
  log('\n🔐 Testing with MOCK authentication mode...', colors.yellow);
  await testEndpoint('User Info (Mock Mode)', 'GET', '/api/v1/user', undefined, 'mock-token');

  // Test 5: Create chat with full access token
  log('\n🔐 Testing with full access JWT token...', colors.yellow);
  const fullAccessToken = mockTokens.fullAccess;

  const chat = await testTRPC('chats.create', {
    title: 'Test Chat with Markdown',
    feature: 'narodne-novine',
    visibility: 'private',
    ragConfig: {
      language: 'hr',
      maxDocuments: 5,
      minConfidence: 0.7,
      temperature: 0.7,
    },
    metadata: {
      tags: ['test', 'markdown'],
      description: 'Testing chat with **Markdown** support',
    },
  }, fullAccessToken);

  // Test 6: List chats
  await testTRPC('chats.list', {
    feature: 'narodne-novine',
    limit: 10,
  }, fullAccessToken, 'query');

  // Test 7: Send message with Markdown
  if (chat?.result?.data?.id) {
    const chatId = chat.result.data.id;

    await testTRPC('messages.send', {
      chatId,
      content: `### Test Question\n\nThis is a test message with **bold**, *italic*, and a list:\n\n1. First item\n2. Second item\n3. Third item\n\n| Column | Value |\n|--------|-------|\n| Test | 123 |\n\n✅ Test complete!`,
      ragConfig: {
        maxDocuments: 5,
        language: 'hr',
      },
    }, fullAccessToken);

    // Test 8: List messages
    await testTRPC('messages.list', {
      chatId,
      limit: 10,
    }, fullAccessToken, 'query');
  }

  // Test 9: Try with read-only token (should fail on create)
  log('\n🔐 Testing with read-only JWT token...', colors.yellow);
  await testTRPC('chats.create', {
    title: 'Should Fail',
    feature: 'narodne-novine',
  }, mockTokens.readOnly);

  // Test 10: Try with no features token
  log('\n🔐 Testing with no-features JWT token...', colors.yellow);
  await testTRPC('chats.create', {
    title: 'Should Fail - No Features',
    feature: 'narodne-novine',
  }, mockTokens.noFeatures);

  log('\n✨ Test suite complete!', colors.green);
}

// Check if API is running
async function checkAPIHealth() {
  try {
    const response = await fetch(`${BASE_URL}/api/v1/health`);
    if (response.ok) {
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

// Main execution
(async () => {
  const isHealthy = await checkAPIHealth();

  if (!isHealthy) {
    log('❌ API is not running. Please start it with: npm run dev', colors.red);
    process.exit(1);
  }

  await runTests();
})();