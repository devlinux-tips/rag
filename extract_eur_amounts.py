#!/usr/bin/env python3
"""
Extract specific EUR amounts from Croatian documents for July 1, 2025 decisions
"""
import re
from pathlib import Path
from src.preprocessing.extractors import DocumentExtractor

def extract_eur_amounts():
    """Extract and display EUR amounts from Croatian documents."""
    
    print("💰 Extracting EUR amounts from Croatian documents")
    print("=" * 50)
    
    extractor = DocumentExtractor()
    
    doc_path = Path("./data/raw")
    documents = list(doc_path.glob("*.pdf")) + list(doc_path.glob("*.docx"))
    
    all_eur_amounts = []
    
    for doc_file in documents:
        try:
            print(f"\n📄 Analyzing {doc_file.name}...")
            
            text = extractor.extract_text(doc_file)
            
            # Check if document contains July 2025 content
            if "srpnja 2025" in text or "srpanj 2025" in text or "1. srpnja 2025" in text:
                print(f"✅ Found July 2025 content")
                
                # Enhanced EUR pattern matching
                eur_patterns = [
                    r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*EUR',  # Format: 1.000.000,00 EUR
                    r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*EUR',      # Format: 1.000.000,00 EUR  
                    r'EUR\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)', # Format: EUR 1.000.000,00
                    r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*€',   # Format: 1.000.000,00 €
                    r'€\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',   # Format: € 1.000.000,00
                    r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*eur', # lowercase eur
                    r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*eura' # Croatian plural form
                ]
                
                found_amounts = []
                
                for pattern in eur_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        amount = match.group(1)
                        start_pos = max(0, match.start() - 100)
                        end_pos = min(len(text), match.end() + 100)
                        context = text[start_pos:end_pos].replace('\n', ' ').strip()
                        
                        found_amounts.append({
                            'amount': amount,
                            'context': context,
                            'source': doc_file.name
                        })
                
                if found_amounts:
                    print(f"   Found {len(found_amounts)} EUR amounts:")
                    for amt in found_amounts:
                        print(f"   💰 {amt['amount']} EUR")
                        print(f"      Context: ...{amt['context'][:150]}...")
                        print()
                    
                    all_eur_amounts.extend(found_amounts)
                else:
                    print(f"   No EUR amounts found")
            else:
                print(f"   No July 2025 content found")
                
        except Exception as e:
            print(f"   ❌ Error processing {doc_file.name}: {e}")
    
    # Summary
    print("\n📊 SUMMARY - EUR Amounts from July 1, 2025 decisions:")
    print("=" * 60)
    
    if all_eur_amounts:
        # Group by amount for cleaner display
        amount_groups = {}
        for item in all_eur_amounts:
            amount = item['amount']
            if amount not in amount_groups:
                amount_groups[amount] = []
            amount_groups[amount].append(item)
        
        for amount, items in sorted(amount_groups.items()):
            print(f"\n💰 Amount: {amount} EUR")
            for item in items:
                print(f"   📄 Source: {item['source']}")
                # Clean context
                clean_context = re.sub(r'\s+', ' ', item['context']).strip()
                print(f"   📝 Context: {clean_context[:200]}...")
        
        print(f"\n🔢 Total unique EUR amounts found: {len(amount_groups)}")
        print(f"🔢 Total EUR references found: {len(all_eur_amounts)}")
        
        # Extract specific July 1, 2025 decisions
        july_1_amounts = []
        for item in all_eur_amounts:
            if "1. srpnja 2025" in item['context']:
                july_1_amounts.append(item)
        
        if july_1_amounts:
            print(f"\n🎯 Specific to July 1, 2025 decisions:")
            for item in july_1_amounts:
                print(f"   💰 {item['amount']} EUR - {item['source']}")
        
    else:
        print("❌ No EUR amounts found in documents with July 2025 content")

if __name__ == "__main__":
    extract_eur_amounts()