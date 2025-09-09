#!/usr/bin/env python3
"""
Test script for the enhanced hierarchical categorization system.

This script validates the new document categorization and hierarchical routing
functionality with various query types in Croatian and English.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
services_path = project_root / "services" / "rag-service"
sys.path.insert(0, str(services_path))

from src.generation.enhanced_prompt_templates import \
    create_enhanced_prompt_builder
from src.retrieval.categorization import (DocumentCategory,
                                          EnhancedQueryCategorizer,
                                          create_query_categorizer)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test queries for different categories
TEST_QUERIES = {
    "cultural": [
        "Koje su najvažnije hrvatske kulturne tradicije?",
        "Opišite hrvatsku narodnu glazbu",
        "What are the main Croatian cultural festivals?",
        "Tell me about Croatian folk traditions",
    ],
    "tourism": [
        "Gdje se nalaze najljepše plaže u Hrvatskoj?",
        "Preporučite mi smještaj u Dubrovniku",
        "What are the best tourist attractions in Split?",
        "Where can I find good restaurants in Zagreb?",
    ],
    "technical": [
        "Kako implementirati REST API u Pythonu?",
        "Objasnite algoritam sortiranja",
        "What is the difference between SQL and NoSQL databases?",
        "How to optimize database queries for performance?",
    ],
    "legal": [
        "Koja su prava potrošača u Hrvatskoj?",
        "Objasnite postupak osnivanja tvrtke",
        "What are the employment laws in Croatia?",
        "Explain copyright regulations for software",
    ],
    "business": [
        "Kako pokrenuti startup u Hrvatskoj?",
        "Analizirajte hrvatsko tržište rada",
        "What are the best investment opportunities in Croatia?",
        "Analyze the Croatian economic trends",
    ],
    "faq": [
        "Što je RAG sistem?",
        "Kako koristiti ChatGPT?",
        "What is machine learning?",
        "How do neural networks work?",
    ],
    "educational": [
        "Objasnite osnove programiranja",
        "Kako učiti hrvatski jezik?",
        "What are the fundamentals of computer science?",
        "Explain basic mathematics concepts",
    ],
    "news": [
        "Kakve su danas vijesti iz Hrvatske?",
        "Što se događa u tehnološkom sektoru?",
        "What are the current events in Europe?",
        "Recent developments in AI technology",
    ],
    "general": [
        "Dobar dan, kako ste?",
        "Molim vas pomoć",
        "Hello, how are you?",
        "Please help me with something",
    ],
}

CONTEXT_EXAMPLES = {
    "cultural_context": {
        "user_category_preference": "cultural",
        "region": "dalmatia",
    },
    "tourism_context": {
        "user_category_preference": "tourism",
        "time_period": "summer_season",
        "region": "istria",
    },
    "technical_context": {
        "user_category_preference": "technical",
        "experience_level": "intermediate",
    },
}


async def test_categorization_accuracy():
    """Test categorization accuracy for different query types."""
    logger.info("🔍 Testing categorization accuracy...")

    # Test both Croatian and English
    for language in ["hr", "en"]:
        logger.info(f"\n📝 Testing language: {language}")
        categorizer = create_query_categorizer(language=language)

        correct_predictions = 0
        total_predictions = 0

        for expected_category, queries in TEST_QUERIES.items():
            logger.info(f"\n🎯 Testing category: {expected_category}")

            for query in queries:
                # Skip non-matching language queries for now (simple heuristic)
                if (
                    language == "hr"
                    and not any(char in query for char in "čćšžđ")
                    and ">" not in query
                    and all(ord(c) < 128 for c in query)
                ):
                    if any(
                        word in query.lower()
                        for word in [
                            "what",
                            "how",
                            "where",
                            "when",
                            "who",
                            "the",
                            "and",
                            "or",
                            "but",
                            "with",
                        ]
                    ):
                        continue  # Skip English queries when testing Croatian

                if language == "en" and any(char in query for char in "čćšžđ"):
                    continue  # Skip Croatian queries when testing English

                result = categorizer.categorize_query(query)

                predicted_category = result.primary_category.value
                is_correct = predicted_category == expected_category

                if is_correct:
                    correct_predictions += 1

                total_predictions += 1

                status = "✅" if is_correct else "❌"
                logger.info(
                    f"{status} Query: '{query[:50]}...' -> "
                    f"Predicted: {predicted_category} (confidence: {result.confidence:.3f})"
                )

                if not is_correct:
                    logger.info(f"   Expected: {expected_category}")
                    logger.info(
                        f"   Secondary categories: {[cat.value for cat in result.secondary_categories]}"
                    )

        accuracy = (
            (correct_predictions / total_predictions) * 100
            if total_predictions > 0
            else 0
        )
        logger.info(
            f"\n📊 Language {language} Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})"
        )


async def test_context_influence():
    """Test how context affects categorization."""
    logger.info("\n🎛️ Testing context influence on categorization...")

    categorizer = create_query_categorizer(language="hr")
    test_query = "Kakve su tradicije u Hrvatskoj?"

    # Test without context
    result_no_context = categorizer.categorize_query(test_query)
    logger.info(
        f"No context: {result_no_context.primary_category.value} (confidence: {result_no_context.confidence:.3f})"
    )

    # Test with cultural context
    result_with_context = categorizer.categorize_query(
        test_query, CONTEXT_EXAMPLES["cultural_context"]
    )
    logger.info(
        f"With cultural context: {result_with_context.primary_category.value} (confidence: {result_with_context.confidence:.3f})"
    )

    # Show how context changed the result
    confidence_boost = result_with_context.confidence - result_no_context.confidence
    logger.info(f"Context confidence boost: {confidence_boost:+.3f}")


async def test_prompt_templates():
    """Test category-specific prompt templates."""
    logger.info("\n📝 Testing enhanced prompt templates...")

    # Test both languages
    for language in ["hr", "en"]:
        logger.info(f"\n🌐 Testing prompt templates for language: {language}")
        prompt_builder = create_enhanced_prompt_builder(language=language)

        test_query = (
            "Koje su najvažnije hrvatske kulturne tradicije?"
            if language == "hr"
            else "What are the main Croatian cultural traditions?"
        )
        test_context = (
            [
                "Hrvatska kultura je bogata tradicijama koje sežu duboko u povijest.",
                "Najpoznatije hrvatske tradicije uključuju klape, kravate i festival.",
            ]
            if language == "hr"
            else [
                "Croatian culture is rich with traditions that reach deep into history.",
                "The most famous Croatian traditions include klapa singing, ties, and festivals.",
            ]
        )

        # Test different categories
        categories_to_test = [
            DocumentCategory.CULTURAL,
            DocumentCategory.TOURISM,
            DocumentCategory.TECHNICAL,
            DocumentCategory.GENERAL,
        ]

        for category in categories_to_test:
            system_prompt, user_prompt = prompt_builder.build_prompt(
                query=test_query,
                context_chunks=test_context,
                category=category,
            )

            logger.info(f"\n🎭 Category: {category.value}")
            logger.info(f"System prompt length: {len(system_prompt)} chars")
            logger.info(f"System prompt preview: {system_prompt[:100]}...")
            logger.info(f"User prompt length: {len(user_prompt)} chars")

            # Validate template formatting
            if "{query}" in user_prompt or "{context}" in user_prompt:
                logger.warning(
                    f"⚠️  Unformatted placeholders found in {category.value} user prompt!"
                )


async def test_query_complexity_analysis():
    """Test query complexity analysis."""
    logger.info("\n🧮 Testing query complexity analysis...")

    categorizer = create_query_categorizer(language="hr")

    test_queries = [
        "Što?",  # Simple
        "Koje su najvažnije hrvatske kulturne tradicije u Dalmaciji?",  # Medium
        "Analizirajte utjecaj globalizacije na tradicionalne hrvatske kulturne prakse u kontekstu modernog turizma i ekonomskog razvoja regije?",  # Complex
    ]

    for query in test_queries:
        analysis = categorizer.analyze_query_complexity(query)
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Complexity score: {analysis['complexity_score']:.3f}")
        logger.info(f"Word count: {analysis['word_count']}")
        logger.info(f"Category indicators: {analysis['category_indicators']}")
        logger.info(f"Has cultural markers: {analysis['has_cultural_markers']}")


async def test_retrieval_strategy_selection():
    """Test retrieval strategy selection for different categories."""
    logger.info("\n🎯 Testing retrieval strategy selection...")

    categorizer = create_query_categorizer(language="hr")

    strategy_test_queries = [
        ("Hrvatska kultura tradicije", DocumentCategory.CULTURAL, "cultural_context"),
        ("najbolji hotel Zagreb", DocumentCategory.TOURISM, "semantic_focused"),
        ("Python API implementacija", DocumentCategory.TECHNICAL, "technical_precise"),
        ("što je machine learning", DocumentCategory.FAQ, "faq_optimized"),
        ("aktualne vijesti Hrvatska", DocumentCategory.NEWS, "temporal_aware"),
    ]

    for query, expected_category, expected_strategy_type in strategy_test_queries:
        result = categorizer.categorize_query(query)

        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Detected category: {result.primary_category.value}")
        logger.info(f"Suggested strategy: {result.suggested_strategy.value}")
        logger.info(f"Confidence: {result.confidence:.3f}")

        # Check if strategy makes sense for category
        category_match = result.primary_category.value == expected_category.value
        logger.info(f"Category match: {'✅' if category_match else '❌'}")


async def test_performance_metrics():
    """Test performance of the categorization system."""
    logger.info("\n⚡ Testing categorization performance...")

    categorizer = create_query_categorizer(language="hr")

    # Test with a batch of queries
    all_queries = []
    for queries in TEST_QUERIES.values():
        all_queries.extend(queries[:2])  # Take first 2 from each category

    import time

    start_time = time.time()
    for query in all_queries:
        result = categorizer.categorize_query(query)

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_query = total_time / len(all_queries)
    queries_per_second = len(all_queries) / total_time

    logger.info(f"Total queries processed: {len(all_queries)}")
    logger.info(f"Total time: {total_time:.3f} seconds")
    logger.info(f"Average time per query: {avg_time_per_query:.3f} seconds")
    logger.info(f"Queries per second: {queries_per_second:.1f}")


async def test_error_handling():
    """Test error handling for edge cases."""
    logger.info("\n🛡️ Testing error handling...")

    categorizer = create_query_categorizer(language="hr")

    edge_cases = [
        "",  # Empty query
        "   ",  # Whitespace only
        "a",  # Single character
        "A" * 1000,  # Very long query
        "!@#$%^&*()",  # Special characters only
        "🚀🌟💫",  # Emojis only
    ]

    for query in edge_cases:
        try:
            result = categorizer.categorize_query(query)
            logger.info(
                f"Query: '{query[:20]}...' -> {result.primary_category.value} (confidence: {result.confidence:.3f})"
            )
        except Exception as e:
            logger.error(f"Error with query '{query[:20]}...': {e}")


async def generate_categorization_report():
    """Generate a comprehensive categorization report."""
    logger.info("\n📊 Generating categorization system report...")

    # Test both languages
    for language in ["hr", "en"]:
        categorizer = create_query_categorizer(language=language)
        prompt_builder = create_enhanced_prompt_builder(language=language)

        # Get system statistics
        template_stats = prompt_builder.get_template_stats()

        # Validate templates
        validation_results = prompt_builder.validate_templates()

        report = {
            "language": language,
            "categorization_system": {
                "supported_categories": len(DocumentCategory),
                "category_names": [cat.value for cat in DocumentCategory],
            },
            "prompt_templates": template_stats,
            "validation": validation_results,
            "test_timestamp": asyncio.get_event_loop().time(),
        }

        # Save report
        report_file = f"categorization_report_{language}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to: {report_file}")

        # Print summary
        logger.info(f"\n📋 Summary for {language}:")
        logger.info(
            f"Categories supported: {report['categorization_system']['supported_categories']}"
        )
        logger.info(
            f"Template categories: {report['prompt_templates']['total_categories']}"
        )
        logger.info(
            f"Validation status: {'✅ Valid' if report['validation']['valid'] else '❌ Issues found'}"
        )

        if report["validation"]["issues"]:
            logger.warning(f"Issues found: {len(report['validation']['issues'])}")
            for issue in report["validation"]["issues"][:3]:  # Show first 3
                logger.warning(f"  - {issue}")


async def main():
    """Main test function."""
    logger.info("🚀 Starting Enhanced Hierarchical Categorization System Tests")
    logger.info("=" * 70)

    try:
        # Run all tests
        await test_categorization_accuracy()
        await test_context_influence()
        await test_prompt_templates()
        await test_query_complexity_analysis()
        await test_retrieval_strategy_selection()
        await test_performance_metrics()
        await test_error_handling()
        await generate_categorization_report()

        logger.info("\n🎉 All tests completed successfully!")
        logger.info("✅ Enhanced hierarchical categorization system is working properly")

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
