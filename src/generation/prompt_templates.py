"""
Prompt templates for Croatian RAG system using local LLM.
Contains system prompts and templates for different query types.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    system_prompt: str
    user_template: str
    context_template: str = "Context:\n{context}\n\n"


class CroatianRAGPrompts:
    """Collection of prompt templates for Croatian RAG system."""
    
    # Base system prompt for Croatian language support
    BASE_SYSTEM_PROMPT = """Ti si korisni asistent koji odgovara na pitanja na hrvatskom jeziku. 
Koristi kontekst iz dokumenata da pružiš točne i korisne odgovore. 
Ako informacije nisu dostupne u kontekstu, jasno to naglasi.
Odgovori uvijek na hrvatskom jeziku, čak i ako je pitanje postavljeno na drugom jeziku."""

    # Template for general question answering
    QUESTION_ANSWERING = PromptTemplate(
        system_prompt=BASE_SYSTEM_PROMPT,
        user_template="Pitanje: {query}\n\nOdgovor:",
        context_template="Kontekst iz dokumenata:\n{context}\n\n"
    )
    
    # Template for document summarization
    SUMMARIZATION = PromptTemplate(
        system_prompt="""Ti si korisni asistent koji stvara sažetke dokumenata na hrvatskom jeziku.
Tvoj zadatak je da napraviš kratak, ali sveobuhvatan sažetak glavnih točaka iz danog teksta.
Fokusiraj se na najvažnije informacije i zaključke.""",
        user_template="Molimo napravi sažetak sljedećeg sadržaja:\n\nSažetak:",
        context_template="Sadržaj za sažetak:\n{context}\n\n"
    )
    
    # Template for factual questions
    FACTUAL_QA = PromptTemplate(
        system_prompt="""Ti si korisni asistent koji odgovara na činjenična pitanja na hrvatskom jeziku.
Koristi samo informacije iz priloženog konteksta. 
Ako odgovor nije dostupan u kontekstu, reci "Ne mogu pronaći tu informaciju u dostupnim dokumentima."
Budi precizan i navedi relevantne detalje.""",
        user_template="Činjenično pitanje: {query}\n\nOdgovor:",
        context_template="Relevantne informacije:\n{context}\n\n"
    )
    
    # Template for explanatory questions
    EXPLANATORY = PromptTemplate(
        system_prompt="""Ti si korisni asistent koji objašnjava složene teme na hrvatskom jeziku.
Koristi kontekst da daš detaljno objašnjenje. 
Objasni pojmove jednostavno i jasno, koristi primjere ako je to moguće.
Struktura odgovor logički s glavnim točkama.""",
        user_template="Molimo objasni: {query}\n\nObjašnjenje:",
        context_template="Relevantni sadržaj:\n{context}\n\n"
    )
    
    # Template for comparison questions
    COMPARISON = PromptTemplate(
        system_prompt="""Ti si korisni asistent koji poredi koncepte i ideje na hrvatskom jeziku.
Koristi kontekst da napraviš detaljnu usporedbu.
Istaknuti sličnosti i razlike, te navedi specifične primjere iz konteksta.""",
        user_template="Usporedi sljedeće: {query}\n\nUsporedba:",
        context_template="Informacije za usporedbu:\n{context}\n\n"
    )


class PromptBuilder:
    """Builder class for constructing prompts from templates and context."""
    
    def __init__(self, template: PromptTemplate):
        """
        Initialize prompt builder with template.
        
        Args:
            template: PromptTemplate to use for building prompts
        """
        self.template = template
    
    def build_prompt(
        self, 
        query: str, 
        context: Optional[List[str]] = None,
        max_context_length: int = 2000
    ) -> tuple[str, str]:
        """
        Build complete prompt from query and context.
        
        Args:
            query: User query
            context: List of context chunks
            max_context_length: Maximum length of context text
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Build context string if provided
        context_text = ""
        if context:
            context_text = self._format_context(context, max_context_length)
        
        # Build user prompt
        user_prompt = ""
        if context_text:
            user_prompt += self.template.context_template.format(context=context_text)
        
        user_prompt += self.template.user_template.format(query=query)
        
        return self.template.system_prompt, user_prompt
    
    def _format_context(self, context: List[str], max_length: int) -> str:
        """
        Format context chunks into single text with length limit.
        
        Args:
            context: List of context chunks
            max_length: Maximum total length
            
        Returns:
            Formatted context text
        """
        if not context:
            return ""
        
        formatted_chunks = []
        total_length = 0
        
        for i, chunk in enumerate(context, 1):
            # Add chunk header
            chunk_header = f"[Dokument {i}]\n"
            chunk_text = chunk_header + chunk.strip() + "\n"
            
            # Check if adding this chunk exceeds limit
            if total_length + len(chunk_text) > max_length:
                if not formatted_chunks:  # At least include first chunk
                    # Truncate the chunk to fit
                    remaining_length = max_length - len(chunk_header) - 10
                    truncated_chunk = chunk[:remaining_length] + "..."
                    formatted_chunks.append(chunk_header + truncated_chunk)
                break
            
            formatted_chunks.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(formatted_chunks)


def get_prompt_for_query_type(query: str) -> PromptTemplate:
    """
    Select appropriate prompt template based on query characteristics.
    
    Args:
        query: User query text
        
    Returns:
        Most suitable PromptTemplate
    """
    query_lower = query.lower()
    
    # Keywords for different query types
    summary_keywords = ['sažetak', 'sažmi', 'ukratko', 'glavne točke', 'resume']
    comparison_keywords = ['usporedi', 'razlika', 'sličnost', 'vs', 'nasuprot']
    explanation_keywords = ['objasni', 'što je', 'kako', 'zašto', 'definiraj']
    factual_keywords = ['kada', 'gdje', 'tko', 'koliko', 'koji', 'koja']
    
    # Check for summary request
    if any(keyword in query_lower for keyword in summary_keywords):
        return CroatianRAGPrompts.SUMMARIZATION
    
    # Check for comparison request
    if any(keyword in query_lower for keyword in comparison_keywords):
        return CroatianRAGPrompts.COMPARISON
    
    # Check for explanation request
    if any(keyword in query_lower for keyword in explanation_keywords):
        return CroatianRAGPrompts.EXPLANATORY
    
    # Check for factual questions
    if any(keyword in query_lower for keyword in factual_keywords):
        return CroatianRAGPrompts.FACTUAL_QA
    
    # Default to general question answering
    return CroatianRAGPrompts.QUESTION_ANSWERING


def create_prompt_builder(query: str) -> PromptBuilder:
    """
    Factory function to create prompt builder for specific query.
    
    Args:
        query: User query
        
    Returns:
        PromptBuilder with appropriate template
    """
    template = get_prompt_for_query_type(query)
    return PromptBuilder(template)