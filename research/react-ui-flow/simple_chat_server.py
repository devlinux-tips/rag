#!/usr/bin/env python3
"""
Simple Chat API Server for Testing React UI
A minimal FastAPI server that mimics the RAG chat API for testing purposes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uuid
from datetime import datetime

app = FastAPI(title="Simple Chat API", version="1.0.0")

# Enable CORS for React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model matching the React app expectations
class SendMessageRequest(BaseModel):
    conversation_id: str
    message: str
    language: str = "hr"
    tenant_slug: str = "development"
    user_id: str = "dev_user"
    max_rag_results: int = 3

# Response model
class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    model: str = "qwen2.5:7b-instruct"
    provider: str = "ollama"
    usage: Dict[str, int] = {"input_tokens": 8, "output_tokens": 50, "total_tokens": 58}

@app.get("/")
async def root():
    return {"message": "Simple Chat API Server", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat/message", response_model=ChatResponse)
async def send_message(request: SendMessageRequest):
    """Mock endpoint that returns the actual RAG response from the example JSON."""

    # Return the actual Croatian RAG response content from the example
    content = """Na dan **1. srpnja 2025. godine** u Republici Hrvatskoj stupaju na snagu i važe sljedeće važne pravne, ekonomske i administrativne mjere, koje su utemeljene na odlukama i zakonskim akcijama donesenim u travnju i kolovozu 2025. godine:

---

### 1. **POVEĆANJE NAJNIŽE MIROVINE ZA JEDNU GODINU MIROVINSKOG STAŽA**
Na osnovi **Odluke Vlade Republike Hrvatske o najnižoj mirovini za jednu godinu mirovinskog staža od 1. srpnja 2025.**, donesene na sjednici 26. kolovoza 2025. (Urbroj: 341-99-01/01-25-7), od 1. srpnja 2025. godine:

- **Najniža mirovina za jednu godinu mirovinskog staža iznosi 15,32 EUR**.
- Ova iznosi se primjenjuje unutar **sustava državne mirovinske osiguranja**, a određuje se ovisno o:
 - Ukupnom broju godina mirovinskog staža,
 - Polaznom faktoru iz članka 83. Zakona o mirovinskom osiguranju,
 - Mirovinskom faktoru iz članka 85. tog zakona.
- Cilj je osigurati minimalnu zaštitu mirovinskih prava osoba s kraćim stažem, u skladu s **Strategijom upravljanja državnom imovinom 2019.–2025.** i zakonom o upravljanju nekretninama i pokretninama u vlasništvu RH.

---

### 2. **OZNAKA DANA ZA UPRAVLJANJE DRŽAVNOM IMOVINOM**
Na temelju **članka 64. stavka 1. Zakona o upravljanju nekretninama i pokretninama u vlasništvu Republike Hrvatske (»Narodne novine« br. 155/23.)**, Vlada je na sjednici 21. kolovoza 2025. donijela odluku:

- **Poduzimanje svih aktivnosti radi ostvarivanja mjera i ciljeva iz strategije upravljanja državnom imovinom za razdoblje 2019.–2025.**
- Od 1. srpnja 2025. godine uključuju se dodatne akcije u:
 - Privatizaciji nekretnina u vlasništvu države,
 - Upravljanju stambenim i poslovnim prostorima,
 - Optimalnom korištenju državne imovine kako bi se povećala efikasnost i prihodi države.

---

### 3. **UPRAVLJANJE VODNIM PODRUČJIMA – NOVE GRANICE**
Na temelju **Uredbe Vlade Republike Hrvatske o granicama vodnih područja**, donesene 28. kolovoza 2025. (na osnovi članka 34. stavka 3. Zakona o vodama – »Narodne novine« br. 66/19., 84/21. i 47/23.), od 1. srpnja 2025. godine:

- Postavljene su nove **granične linije vodnih područja**, uključujući:
 - Granice vodnih područja na teritoriju reka (npr. Drava, Kupa, Sava),
 - Područja izvora i vodnih rezervoara,
 - Područja zaštite vodnih izvora i ekološke nesmetanosti.
- Ova uredba ima cilj:
 - Ujednačiti pravne okvire za upravljanje vodnim resursima,
 - Omogućiti bolju zaštitu okoliša,
 - Povećati transparentnost u upravljanju vodama.

---

### 4. **INFORMACIJE O ODREĐIVANJU NAKNADA ZA TJELESNO OŠTEĆENJE**
Na osnovi **Odluke o osnovici za određivanje naknade zbog tjelesnog oštećenja i o usklađivanju novčanih naknada od 1. srpnja 2025.**, donesene 26. kolovoza 2025., od 1. srpnja:

- Uvedena je **novčana osnovica za izračun naknade zbog tjelesnog oštećenja**.
- Naknade se usklađuju s inflacijom i stvarnim ekonomskim uvjetima, uz obvezu **prilagodbe svake godine**.
- Ova mjera utječe na:
 - Odluke u sudskim postupcima za ozbiljna tjelesna oštećenja,
 - Iznos naknada u slučajevima nezgoda na radu, prometnih nesreća, kaznenih djela, itd.

---

### 5. **DODATNE POSLOVNE I ADMINISTRACIJSKE PROMJENE**
- **Hrvatski zavod za mirovinsko osiguranje (HZMO)** primjenjuje nove računske metode za izračun mirovina, posebno u slučajevima kombiniranih mirovina i prilika za uključenje u "mjerila za osigurane".
- Uvod se **digitalne tranzakcije** u upravljanju mirovinskim podacima, s ciljem povećanja sigurnosti i brzine obrade zahtjeva.

---

### ZAKLJUČAK
Od **1. srpnja 2025. godine** u Republici Hrvatskoj:
- **Najniža mirovina za jednu godinu staža iznosi 15,32 EUR**,
- Uvedene su nove **granične linije vodnih područja**,
- Uspostavljena je nova **osnovica za naknade zbog tjelesnog oštećenja**,
- Uspostavljena je **strategijska uprava državnom imovinom** s ciljem efikasnijeg korištenja državnih resursa. Sve ove mjere imaju za cilj **povećanje pravne sigurnosti, ekonomske stabilnosti i društvene pravde** u Hrvatskoj, u skladu s nacionalnim strategijama i Europskom unije.

---

👉 *Izvor: Narodne novine – broj 155/23., 96/19., 66/19., 84/21., 47/23., odluke i uredba Vlade RH iz kolovoza 2025. (Urbroj: 341-99-01/01-25-7, 341-99-01/01-25-8, itd.).*"""

    # Generate response matching the original RAG response format
    response = ChatResponse(
        conversation_id=request.conversation_id,
        message_id=f"rag-{uuid.uuid4().hex[:8]}",
        content=content,
        model="qwen/qwen3-30b-a3b-instruct-2507",
        provider="openrouter",
        usage={
            "input_tokens": len(request.message.split()),
            "output_tokens": 594,
            "total_tokens": len(request.message.split()) + 594
        }
    )

    return response

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Chat API Server...")
    print("📱 React Frontend: http://localhost:3000")
    print("🔧 API Docs: http://localhost:8080/docs")
    print("🛑 Press Ctrl+C to stop")

    uvicorn.run(
        "simple_chat_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )