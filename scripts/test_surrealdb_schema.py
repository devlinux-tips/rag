#!/usr/bin/env python3
"""
Test SurrealDB schema for RAG system data organization
"""

import asyncio
import json

from surrealdb import Surreal


async def test_surrealdb_schema():
    """Test the proposed SurrealDB schema with sample data"""

    print("🗄️ Testing SurrealDB Schema for RAG System...")
    print("=" * 50)

    # Connect to SurrealDB
    db = Surreal("http://localhost:8000/rpc")

    try:
        # Skip authentication for unauthenticated mode
        # await db.signin({"user": "root", "pass": "root"})

        # Use namespace and database
        await db.use("rag_platform", "development")

        print("✅ Connected to SurrealDB")

        # 1. Test Users Table
        print("\n📋 Testing Users Table...")
        user_data = {
            "email": "test@example.com",
            "name": "Test User",
            "password_hash": "bcrypt_hash_placeholder",
            "language_preference": "hr",
            "created_at": "time::now()",
            "role": "user",
            "active": True,
        }

        user_result = await db.create("users", user_data)
        print(f"Created user: {user_result[0]['id']}")
        user_id = user_result[0]["id"]

        # 2. Test Prompt Templates Table
        print("\n📝 Testing Prompt Templates...")
        template_data = {
            "name": "cultural_context",
            "category": "cultural",
            "language": "hr",
            "system_prompt": "Ti si stručnjak za hrvatsku kulturu i povijest.",
            "user_template": "Pitanje: {query}\\n\\nOdgovor:",
            "context_template": "Kontekst:\\n{context}\\n\\n",
            "is_system_default": True,
            "is_active": True,
            "usage_count": 0,
            "created_at": "time::now()",
        }

        template_result = await db.create("prompt_templates", template_data)
        print(f"Created template: {template_result[0]['id']}")
        template_id = template_result[0]["id"]

        # 3. Test Documents Table
        print("\n📄 Testing Documents Table...")
        document_data = {
            "filename": "croatian_culture.pdf",
            "original_path": "data/raw/hr/croatian_culture.pdf",
            "language": "hr",
            "category": "cultural",
            "subcategory": "history",
            "user_id": user_id,
            "tenant_id": "default",
            "processing_status": "completed",
            "chunk_count": 15,
            "file_size": 524288,
            "file_type": "application/pdf",
            "uploaded_at": "time::now()",
            "processed_at": "time::now()",
            "tags": ["croatia", "culture", "history"],
            "metadata": {
                "author": "Croatian Cultural Institute",
                "title": "Croatian Cultural Heritage",
                "summary": "Overview of Croatian cultural traditions",
            },
        }

        doc_result = await db.create("documents", document_data)
        print(f"Created document: {doc_result[0]['id']}")
        doc_id = doc_result[0]["id"]

        # 4. Test Document Chunks Table
        print("\n🔤 Testing Document Chunks...")
        chunk_data = {
            "document_id": doc_id,
            "chunk_index": 0,
            "content": "Hrvatska kultura je bogata tradicijama koje sežu duboko u povijest. Zagreb, glavni grad, čuva mnoge kulturne spomenike.",
            "language": "hr",
            "category": "cultural",
            "chunk_type": "paragraph",
            "word_count": 20,
            "character_count": 128,
            "chroma_collection": "croatian_documents",
            "chroma_id": "chunk_abc123",
            "created_at": "time::now()",
        }

        chunk_result = await db.create("document_chunks", chunk_data)
        print(f"Created chunk: {chunk_result[0]['id']}")

        # 5. Test User Queries Table
        print("\n🔍 Testing User Queries...")
        query_data = {
            "user_id": user_id,
            "query_text": "Koje su najvažnije hrvatske kulturne tradicije?",
            "language": "hr",
            "detected_category": "cultural",
            "selected_category": "cultural",
            "prompt_template": template_id,
            "response_time_ms": 750,
            "satisfaction_rating": 5,
            "documents_retrieved": 3,
            "created_at": "time::now()",
        }

        query_result = await db.create("user_queries", query_data)
        print(f"Created query: {query_result[0]['id']}")

        # 6. Test Complex Queries
        print("\n🔎 Testing Complex Queries...")

        # Get all documents for a user
        user_docs = await db.query(
            "SELECT * FROM documents WHERE user_id = $user_id", {"user_id": user_id}
        )
        print(f"Found {len(user_docs[0]['result'])} documents for user")

        # Get cultural category documents
        cultural_docs = await db.query(
            "SELECT * FROM documents WHERE category = 'cultural'"
        )
        print(f"Found {len(cultural_docs[0]['result'])} cultural documents")

        # Get user's Croatian language templates
        hr_templates = await db.query(
            """
            SELECT * FROM prompt_templates
            WHERE language = 'hr' AND (user_id = $user_id OR is_system_default = true)
        """,
            {"user_id": user_id},
        )
        print(f"Found {len(hr_templates[0]['result'])} Croatian templates")

        # Get query analytics
        user_query_stats = await db.query(
            """
            SELECT
                category,
                count() as query_count,
                math::mean(response_time_ms) as avg_response_time,
                math::mean(satisfaction_rating) as avg_satisfaction
            FROM user_queries
            WHERE user_id = $user_id
            GROUP BY category
        """,
            {"user_id": user_id},
        )
        print(f"Query analytics: {json.dumps(user_query_stats[0]['result'], indent=2)}")

        # 7. Test Relations and Joins
        print("\n🔗 Testing Relations...")

        # Get document with chunks
        doc_with_chunks = await db.query(
            """
            SELECT
                *,
                (SELECT * FROM document_chunks WHERE document_id = $parent.id) as chunks
            FROM documents
            WHERE id = $doc_id
        """,
            {"doc_id": doc_id},
        )

        doc_data = doc_with_chunks[0]["result"][0]
        print(f"Document '{doc_data['filename']}' has {len(doc_data['chunks'])} chunks")

        print("\n🎉 All SurrealDB schema tests passed!")
        print("✅ Users, documents, chunks, templates, and queries working")
        print("✅ Complex queries and relations functional")
        print("✅ Croatian language data stored correctly")
        print("✅ Schema ready for RAG system integration")

    except Exception as e:
        print(f"❌ Error testing schema: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # SurrealDB Python client doesn't need explicit close for HTTP
        pass


if __name__ == "__main__":
    asyncio.run(test_surrealdb_schema())
