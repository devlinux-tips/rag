














These are my so far favourites, now that you know them, can you do one suprise batch?
Kognytiv
Synaptyx
Qyntex
Semantyx
Neurex
Cerebryx
eoan
QÆON
XAE0N
Serenity
Anubis
ALTAIR
Solaris
Solarix
Titanx
Nexros
EREBUS
Proxima
Arcturus

Synaptyx, Semantyx, Nexros, PYRAXIS



## 🧪 Terminal Usage Examples for Multilingual RAG System

### **1. Health Check & System Status**

```bash
# Check system health for Croatian
python -m src.pipeline.rag_system --lang hr --health

# Check system health for English
python -m src.pipeline.rag_system --lang en --health

# Get system statistics
python -m src.pipeline.rag_system --lang hr --stats
```

### **2. Adding Documents**

```bash
# Add Croatian documents (you already have these)
python -m src.pipeline.rag_system --lang hr --add-docs data/raw/hr/*.pdf data/raw/hr/*.docx

# Add English documents (need to create some first)
python -m src.pipeline.rag_system --lang en --add-docs data/raw/en/*.pdf data/raw/en/*.txt

# Add specific Croatian document
python -m src.pipeline.rag_system --lang hr --add-docs "data/raw/hr/110 - 11.docx"
```

### **3. Croatian Query Examples**

```bash
# Basic Croatian query
python -m src.pipeline.rag_system --lang hr --query "Što piše o novčanim iznosima?"

# Legal document query in Croatian
python -m src.pipeline.rag_system --lang hr --query "Koje su glavne obveze i prava u ovom dokumentu?"

# Specific information query
python -m src.pipeline.rag_system --lang hr --query "Koliki su troškovi i rokovi navedeni u dokumentu?"

# Complex analytical query
python -m src.pipeline.rag_system --lang hr --query "Analiziraj ključne točke i daj sažetak najvažnijih informacija"
```

### **4. English Query Examples**

```bash
# Basic English query
python -m src.pipeline.rag_system --lang en --query "What are the main points discussed in the documents?"

# Technical query
python -m src.pipeline.rag_system --lang en --query "What technical specifications or requirements are mentioned?"

# Analytical query
python -m src.pipeline.rag_system --lang en --query "Summarize the key findings and recommendations"

# Specific search
python -m src.pipeline.rag_system --lang en --query "What costs and timelines are specified?"
```
