"""
Integration tests for the complete preprocessing pipeline.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append("src")

from preprocessing import chunk_croatian_document, extract_document_text
from preprocessing.chunkers import DocumentChunker
from preprocessing.cleaners import CroatianTextCleaner
from preprocessing.extractors import DocumentExtractor


class TestPreprocessingPipelineIntegration:
    """Test the complete preprocessing pipeline integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_text = """
        VLADA REPUBLIKE HRVATSKE
        Zagreb, 30. srpnja 2025.

        ODLUKA
        o produljenju roka za utrošak namjenskih sredstava

        Na temelju članka 11. stavka 4. Zakona o obnovi zgrada oštećenih potresom
        na području Grada Zagreba, Krapinsko-zagorske županije, Zagrebačke županije
        i Sisačko-moslavačke županije („Narodne novine", br. 102/20., 114/21.,
        79/22. i 111/22.), Vlada Republike Hrvatske je na sjednici održanoj
        24. listopada 2024. donijela sljedeću

        O D L U K U

        Članak 1.

        Produljuje se rok za utrošak namjenskih sredstava planiranih u Državnom
        proračunu Republike Hrvatske za 2025. godinu radi financiranja obnove
        zgrada oštećenih potresom od 5. listopada 2024. na području Općine Podgora.
        """.strip()

        self.short_text = "Kratki hrvatski tekst sa čćžšđ."

    def test_full_pipeline_with_mock_document(self):
        """Test complete pipeline from extraction to chunking."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp_file:
            tmp_file.write(self.test_text)
            tmp_path = Path(tmp_file.name)

        try:
            # Step 1: Extract text
            extracted_text = extract_document_text(tmp_path)
            assert len(extracted_text) > 0
            assert "VLADA REPUBLIKE HRVATSKE" in extracted_text
            assert "čćžšđ" in extracted_text or "oštećenih" in extracted_text

            # Step 2: Chunk the text
            chunks = chunk_croatian_document(
                text=extracted_text,
                source_file=str(tmp_path),
                chunk_size=300,
                overlap=50,
                strategy="sliding_window",
            )

            assert len(chunks) > 0
            assert all(chunk.source_file == str(tmp_path) for chunk in chunks)
            assert all(len(chunk.content) > 0 for chunk in chunks)

            # Verify Croatian text is preserved
            combined_content = " ".join(chunk.content for chunk in chunks)
            assert "VLADA REPUBLIKE HRVATSKE" in combined_content

        finally:
            tmp_path.unlink()  # Clean up

    def test_pipeline_with_different_strategies(self):
        """Test pipeline with different chunking strategies."""
        strategies = ["sliding_window", "sentence", "paragraph"]

        results = {}

        for strategy in strategies:
            chunks = chunk_croatian_document(
                text=self.test_text,
                source_file="test.txt",
                chunk_size=200,
                overlap=40,
                strategy=strategy,
            )

            results[strategy] = {
                "chunk_count": len(chunks),
                "avg_length": (sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0),
                "total_chars": sum(len(c.content) for c in chunks),
            }

            # All strategies should produce some chunks
            assert len(chunks) > 0

            # All chunks should have proper metadata
            for chunk in chunks:
                assert chunk.chunk_id.startswith("test")
                assert chunk.source_file == "test.txt"
                assert chunk.word_count > 0
                assert chunk.char_count == len(chunk.content)

        # Compare strategies
        assert len(results) == 3
        # All strategies should process roughly the same amount of text
        total_chars = [results[s]["total_chars"] for s in strategies]
        assert max(total_chars) - min(total_chars) < len(self.test_text) * 0.2

    def test_pipeline_preserves_croatian_characteristics(self):
        """Test that the full pipeline preserves Croatian text characteristics."""
        croatian_sample = """
        Šišmiš je sisavac koji pripada redu Chiroptera. Ovi noćni lovci koriste
        eholociranje za navigaciju i hvatanje plijena. U Hrvatskoj živi nekoliko
        vrsta šišmiša, uključujući veliki potkovnjak i mali potkovnjak.

        Šišmiševi imaju važnu ulogu u ekosustavu jer se hrane insektima, posebno
        komarima i moljevima. Čine važan dio bioraznolikosti naših šuma i gradova.
        """

        chunks = chunk_croatian_document(
            text=croatian_sample,
            source_file="croatian_test.txt",
            chunk_size=150,
            strategy="sentence",
        )

        assert len(chunks) > 0

        # Check Croatian diacritic preservation
        croatian_chars = set("čćžšđČĆŽŠĐ")
        original_chars = croatian_chars.intersection(set(croatian_sample))

        combined_chunks = " ".join(chunk.content for chunk in chunks)
        result_chars = croatian_chars.intersection(set(combined_chunks))

        # Should preserve most Croatian characters
        assert len(result_chars) >= len(original_chars) * 0.8

        # Check specific Croatian words are preserved
        croatian_words = ["šišmiš", "eholociranje", "uključujući", "bioraznolikosti"]
        for word in croatian_words:
            if word.lower() in croatian_sample.lower():
                assert any(word.lower() in chunk.content.lower() for chunk in chunks)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with problematic inputs."""
        # Test with empty text
        chunks = chunk_croatian_document("", "empty.txt")
        assert chunks == []

        # Test with whitespace only
        chunks = chunk_croatian_document("   \n\n   ", "whitespace.txt")
        assert chunks == []

        # Test with very short text
        chunks = chunk_croatian_document("Kratko.", "short.txt", min_chunk_size=50)
        # Should either return empty list or single chunk, depending on filtering
        assert len(chunks) <= 1

    def test_pipeline_performance_with_large_text(self):
        """Test pipeline performance with larger text volumes."""
        # Create larger text sample
        large_text = self.test_text * 20  # Repeat to create ~2000+ character text

        import time

        start_time = time.time()

        chunks = chunk_croatian_document(
            text=large_text,
            source_file="large_test.txt",
            chunk_size=400,
            overlap=50,
            strategy="sliding_window",
        )

        processing_time = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds for this size)
        assert processing_time < 5.0

        # Should produce reasonable number of chunks
        expected_chunks = len(large_text) // 350  # Rough estimate
        assert len(chunks) >= expected_chunks * 0.5  # At least half expected
        assert len(chunks) <= expected_chunks * 2  # Not more than double

    def test_pipeline_component_integration(self):
        """Test that all pipeline components work together correctly."""
        # Create instances of all components
        extractor = DocumentExtractor()
        cleaner = CroatianTextCleaner()
        chunker = DocumentChunker()

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp_file:
            tmp_file.write(self.test_text)
            tmp_path = Path(tmp_file.name)

        try:
            # Test each component in sequence
            # 1. Extract
            raw_text = extractor.extract_text(tmp_path)
            assert len(raw_text) > 0

            # 2. Clean
            cleaned_text = cleaner.clean_text(raw_text)
            assert len(cleaned_text) > 0
            assert len(cleaned_text) <= len(raw_text)  # Should not increase size

            # 3. Chunk
            chunks = chunker.chunk_document(cleaned_text, str(tmp_path))
            assert len(chunks) > 0

            # Verify end-to-end data flow
            total_chunk_chars = sum(len(chunk.content) for chunk in chunks)
            # Should capture most of the cleaned text (accounting for overlaps)
            assert total_chunk_chars >= len(cleaned_text) * 0.8

        finally:
            tmp_path.unlink()


class TestRealWorldScenarios:
    """Test scenarios that mirror real-world usage patterns."""

    def test_government_document_processing(self):
        """Test processing that simulates Croatian government documents."""
        gov_doc = """
        NARODNE NOVINE
        SLUŽBENI LIST REPUBLIKE HRVATSKE
        BROJ 115, ZAGREB, 27. KOLOVOZA 2025.

        GODIŠNJI PLAN UPRAVLJANJA NEKRETNINAMA I POKRETNINAMA
        U VLASNIŠTVU REPUBLIKE HRVATSKE ZA 2026. GODINU

        Na temelju članka 11. stavka 1. Zakona o upravljanju nekretninama
        i pokretninama u vlasništvu Republike Hrvatske („Narodne novine",
        broj 155/23.), Vlada Republike Hrvatske je na sjednici održanoj
        24. srpnja 2025. godine donijela

        O D L U K U

        Članak 1.

        Usvaja se Godišnji plan upravljanja nekretninama i pokretninama
        u vlasništvu Republike Hrvatske za 2026. godinu.
        """

        chunks = chunk_croatian_document(
            text=gov_doc,
            source_file="NN-115-2025.txt",
            chunk_size=250,
            strategy="paragraph",
        )

        assert len(chunks) > 0

        # Should preserve official Croatian terminology
        combined_text = " ".join(chunk.content for chunk in chunks)
        assert "Republike Hrvatske" in combined_text
        assert "Narodne novine" in combined_text or "NARODNE NOVINE" in combined_text

        # Should handle legal document structure
        assert any("Članak" in chunk.content for chunk in chunks)

    def test_mixed_content_document(self):
        """Test document with mixed content types (headers, paragraphs, lists)."""
        mixed_doc = """
        IZVJEŠTAJ O RADU 2024.

        1. UVOD

        Ovaj izvještaj sadrži pregled aktivnosti tijekom 2024. godine.
        Uključuje sljedeće glavne područja:

        • Financijsko poslovanje
        • Upravljanje ljudskim potencijalima
        • Razvoj novih projekata
        • Međunarodna suradnja

        2. FINANCIJSKI REZULTATI

        Financijski rezultati pokazuju pozitivne trendove:

        - Ukupni prihodi: 150.000.000 HRK
        - Ukupni rashodi: 140.000.000 HRK
        - Dobit: 10.000.000 HRK

        3. ZAKLJUČAK

        Godina 2024. bila je uspješna po svim pokazateljima.
        """

        chunks = chunk_croatian_document(
            text=mixed_doc,
            source_file="izvjestaj-2024.txt",
            chunk_size=200,
            strategy="sliding_window",
        )

        assert len(chunks) > 0

        # Should handle different content structures
        content_types = {
            "headers": any(
                "IZVJEŠTAJ" in chunk.content or "UVOD" in chunk.content for chunk in chunks
            ),
            "lists": any("•" in chunk.content or "-" in chunk.content for chunk in chunks),
            "numbers": any("HRK" in chunk.content for chunk in chunks),
        }

        # Should capture different content types
        assert sum(content_types.values()) >= 2

    def test_academic_croatian_text(self):
        """Test processing of academic Croatian text with complex vocabulary."""
        academic_text = """
        SAŽETAK

        Ova disertacija istražuje morfosintaktičke značajke hrvatskoga jezika
        u kontekstu indoeuropskih jezika. Analiza se fokusira na deklinacijske
        paradigme imenica i priloga, posebno na dijakronijski razvoj
        lokativnih konstrukcija.

        Metodologija uključuje korpusnu analizu tekstova iz različitih
        povijesnih razdoblja, od staroslovenskih spomenika do suvremenih
        književnih djela. Rezultati pokazuju značajne promjene u uporabi
        prijedložno-padežnih konstrukcija kroz stoljeća.

        Ključne riječi: morfosintaksa, hrvatski jezik, dijakronijska lingvistika
        """

        chunks = chunk_croatian_document(
            text=academic_text,
            source_file="disertacija.txt",
            chunk_size=180,
            strategy="sentence",
        )

        assert len(chunks) > 0

        # Should preserve complex Croatian academic vocabulary
        complex_terms = [
            "morfosintaktičke",
            "dijakronijski",
            "paradigme",
            "prijedložno-padežnih",
        ]
        combined_text = " ".join(chunk.content for chunk in chunks)

        for term in complex_terms:
            if term in academic_text:
                assert term in combined_text


class TestDataIntegrityAndConsistency:
    """Test data integrity and consistency across the pipeline."""

    def test_text_preservation_through_pipeline(self):
        """Test that essential text content is preserved through the pipeline."""
        original_text = """
        Važne informacije o projektu:

        1. Početak projekta: 1. siječnja 2025.
        2. Završetak projekta: 31. prosinca 2025.
        3. Ukupan proračun: 5.000.000 HRK
        4. Odgovorna osoba: Dr. Ivo Ivić, PhD

        Napomene:
        - Projekt uključuje međunarodnu suradnju
        - Potrebna je suglasnost Ministarstva
        """

        chunks = chunk_croatian_document(
            text=original_text,
            source_file="projekt.txt",
            chunk_size=150,
            strategy="paragraph",
        )

        # Reconstruct text from chunks (simplified)
        reconstructed = " ".join(chunk.content for chunk in chunks)

        # Key information should be preserved
        key_info = ["1. siječnja 2025", "5.000.000 HRK", "Dr. Ivo Ivić", "Ministarstva"]
        for info in key_info:
            if info in original_text:
                assert info in reconstructed

    def test_chunk_metadata_consistency(self):
        """Test that chunk metadata is consistent and accurate."""
        test_text = "Test Croatian text. " * 50  # Create longer text

        chunks = chunk_croatian_document(
            text=test_text,
            source_file="consistency_test.txt",
            chunk_size=100,
            overlap=20,
        )

        for i, chunk in enumerate(chunks):
            # Check chunk index consistency
            assert chunk.chunk_index == i

            # Check character count accuracy
            assert chunk.char_count == len(chunk.content)

            # Check word count accuracy
            expected_word_count = len(chunk.content.split())
            assert chunk.word_count == expected_word_count

            # Check chunk_id format
            assert chunk.chunk_id.startswith("consistency_test")
            assert f"_{i:04d}" in chunk.chunk_id

    def test_no_data_loss_with_overlapping_chunks(self):
        """Test that overlapping chunks don't lose data."""
        source_text = (
            "Rečenica broj jedan. Rečenica broj dva. Rečenica broj tri. Rečenica broj četiri."
        )

        chunks = chunk_croatian_document(
            text=source_text,
            source_file="overlap_test.txt",
            chunk_size=30,
            overlap=15,
            strategy="sliding_window",
        )

        if len(chunks) > 1:
            # Check that overlap exists between adjacent chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]

                # There should be some content overlap or adjacency
                current_end = current_chunk.end_char
                next_start = next_chunk.start_char

                # Should have overlap (next_start < current_end) or be adjacent
                assert next_start <= current_end
