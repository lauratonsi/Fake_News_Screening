# Fake News Screening (HDSS)

[English](README.md) | **Italiano**

Un sistema ibrido di screening della disinformazione: una SVM calibrata, una
Bi-GRU e una Bi-LSTM votano su testi di notizie in inglese, supportate da una
ricerca per similarità *semantica* sui corpora di addestramento e da un flag
di revisione umana quando i modelli sono in disaccordo.
Nato come progetto universitario di IA, qui è stato ricostruito come una
pipeline pulita e riproducibile: **analisi del dataset → modelli → demo
Streamlit**.

Demo live: https://fake-news-screening.streamlit.app/

> **Il dato onesto:** l'ensemble ottiene **94,6%** su un test set in-domain
> senza leakage e **78,1%** su 64 scenari adversarial fuori dominio, che
> spaziano tra claim brevi e articoli lunghi, sei domini tematici e due *stili*
> di disinformazione. Contro la disinformazione classica in stile umano
> (deals segreti, memo trapelati, whistleblower) tiene **zero falsi
> negativi** — nessuna bufala di questo tipo passa mai. Contro un testo
> fluente e attribuito a una fonte senza quei tropi, testato su vere
> riscritture ChatGPT-3.5 di disinformazione documentata (non scritte da
> questo progetto — vedi sotto il perché conta), il recall scende all'**83%**
> — un divario reale ma molto più modesto di quanto suggerisse un primo
> tentativo, metodologicamente viziato, di misurarlo. Quel divario, e come è
> stato misurato, è il dato onesto: vedi *"La disinformazione generata
> dall'IA è più difficile da individuare"* più sotto. Il layer di retrieval
> serve a *trovare* i claim veri/falsi noti più simili, non ad affermare la
> verità partendo da una vicinanza tematica — vedi *"Due usi molto diversi
> degli embeddings"* per capire perché questa distinzione conta e quanto costa.

## Motivazione e contesto di ricerca

Questo strumento nasce da uno studio sulla **sicurezza cognitiva** — la
protezione del giudizio umano dalla manipolazione intenzionale. La premessa è
che la frontiera della sicurezza si è spostata: le moderne operazioni ibride
mirano sempre più a *come le persone decidono*, non alle macchine che usano,
per cui la classica triade CIA (riservatezza, integrità, disponibilità) non
copre più l'intera superficie d'attacco. L'IA generativa acuisce l'asimmetria:
un attore ostile può oggi fabbricare migliaia di varianti persuasive e
linguisticamente native di una notizia falsa a costo marginale quasi nullo,
mentre la verifica resta lenta e costosa.

Due idee di quel lavoro plasmano direttamente il progetto:

- **"Fake news" è l'unità di analisi sbagliata.** Il framework *Information
  Disorder* di Wardle & Derakhshan (Consiglio d'Europa, 2017) distingue
  *misinformation* (falso, condiviso senza intento di nuocere), *disinformation*
  (falso, deliberatamente dannoso) e *malinformation* (contenuto vero usato
  come arma). Un classificatore di testo può toccare solo il segnale di
  falsità-del-contenuto dei primi due — è cieco all'intento, e
  strutturalmente cieco alla malinformation, dove il contenuto è vero. Il
  limite onesto di un sistema così è quindi **screening, non verdetto**:
  vive dentro un processo umano, non al suo posto.
- **Non puoi difendere ciò che non hai testato.** Lo stesso motivo per cui i
  team di sicurezza conducono esercizi adversarial è il motivo per cui gli
  scenari fuori dominio restano nel repo come stress test permanente e
  ripetibile (vedi il benchmark più sotto), non come numero una-tantum — e per
  cui i layer aggiunti sopra il classificatore tendono all'*inoculazione*
  (mostrare le tecniche di manipolazione con cui il lettore viene bersagliato)
  piuttosto che a un mero timbro vero/falso. Lo stesso benchmark è anche dove
  la domanda di ricerca trova una risposta misurabile, per quanto scomoda:
  vedi *"La disinformazione generata dall'IA è più difficile da individuare"*
  più sotto.

La domanda di ricerca originaria era, in sintesi: *cosa può e cosa non può
essere automatizzato nella difesa dello spazio informativo, e dove l'umano deve
restare nel loop?* Questo repository è la metà applicata e misurabile di quella
risposta — un sistema funzionante costruito per essere onesto sui propri limiti.

## Il problema del "99% di accuratezza"

I primi esperimenti sul corpus ISOT portavano *ogni* architettura oltre il
98% di accuratezza. Il notebook
[`notebooks/01_dataset_bias_analysis.ipynb`](notebooks/01_dataset_bias_analysis.ipynb)
documenta perché quei numeri sono un campanello d'allarme, non un risultato:

| Bias nei dati | Effetto |
|---|---|
| **Leakage stilistico** — gli articoli fake hanno in media 2,16 `!`/`?` per articolo e il 30% di maiuscole nei titoli, quelli veri 0,17 e 6% | i modelli imparano la punteggiatura, non il contenuto |
| **Leakage di fonte** — il 99,2% degli articoli "veri" contiene la dicitura `(Reuters)`, lo 0,0% di quelli falsi | l'etichetta è letteralmente scritta nel testo |
| **Cecità temporale** — solo politica USA 2015–2017, con volumi di veri/falsi disallineati nel tempo | tutto ciò che è successivo al 2018 (COVID, elezioni) è fuori dominio |

<p align="center">
  <img src="reports/figures/style_leakage.png" width="48%" alt="Gli articoli fake hanno in media 2,16 punti esclamativi/interrogativi per articolo contro 0,17 dei veri, e il 30% di maiuscole nei titoli contro il 6% dei veri" />
  <img src="reports/figures/reuters_leakage.png" width="48%" alt="Il 99,2% degli articoli veri contiene la dicitura (Reuters) contro lo 0,0% degli articoli falsi" />
</p>
<p align="center">
  <img src="reports/figures/temporal_window.png" width="70%" alt="Volumi mensili di articoli veri e falsi dal 2015 al 2018, che mostrano una finestra temporale stretta e disallineata" />
</p>

## Cosa fa il sistema per contrastarlo

1. **Fusione multi-dataset** — ISOT + WELFake (filtrato per qualità: lunghezza,
   rapporto di maiuscole, punteggiatura) + claim COVID-19, deduplicati: 53.661
   articoli unici.
2. **Protocollo di split rigoroso** — split train/test *prima* di qualunque
   oversampling; la quota COVID è bilanciata e potenziata ×3 solo sul lato
   training; tutti i modelli condividono lo stesso test set intatto (10.733
   articoli). Correggere solo questo protocollo ha spostato la SVM da un
   dichiarato ~98% a un reale 95,3%.
3. **Ensemble di modelli economici e trasparenti** — baseline TF-IDF +
   LinearSVC calibrata, più due RNN bidirezionali leggere (~1,3 MB ciascuna),
   servite come modelli TFLite tramite l'interprete `ai-edge-litert` (~10 MB)
   invece del runtime TensorFlow completo; il punteggio finale è la media
   semplice.
4. **Livello di retrieval di riferimento** — similarità di embeddings di
   frase ([`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
   rispetto a snippet dei ~68k articoli noti come veri/falsi: riconosce un
   claim *riformulato*, non solo letterale. Questo è *retrieval su ciò che
   il sistema ha già visto*, *non* fact-checking, e la demo mostra
   esplicitamente le evidenze recuperate.
5. **Retrieval a livello di claim** — l'input è diviso in frasi simili a
   affermazioni verificabili, e ogni claim viene recuperato in modo
   indipendente, così l'interfaccia può mostrare, per ciascun claim, se
   corrisponde a un'affermazione falsa nota, a un articolo reale noto, o se
   non ha corrispondenze — etichette di *evidenza*, non giudizi di verità.
6. **Retrieval live** — i primi claim vengono controllati anche su fonti live
   gratuite, in ordine di precedenza: Google Fact Check Tools (un vero
   *verdetto* di fact-checking, quando è configurata una chiave API), poi
   Wikipedia (contesto affidabile e senza chiave), con GDELT come ricerca
   notizie di ultima istanza. Un verdetto di fact-checking live ha la
   precedenza per quel claim; altrimenti decide il corpus committato, così il
   sistema funziona anche completamente offline.
7. **Flag di revisione umana** — quando i tre modelli sono in forte
   disaccordo (scarto > 0,40), il verdetto viene segnalato come a bassa
   affidabilità invece di essere presentato come certo.

## Livelli di affidabilità e prebunking

Quattro livelli si aggiungono sopra il classificatore per renderlo utilizzabile
come vero strumento di screening e non una demo — ciascuno scelto per attaccare
una debolezza misurata dei modelli base (vedi il benchmark adversarial: ogni
errore è un falso positivo sicuro su un claim breve *vero*).

8. **Fact-check live come verdetto di prima classe** — il rating di Google Fact
   Check (il verdetto di un fact-checker professionale) ora *domina* l'ensemble
   nel risultato principale, non è più solo un pannello a lato. È l'unico
   segnale insieme autorevole e *attuale*, quindi è ciò che permette allo
   strumento di aver ragione su eventi che i dati di training 2015–2020 non
   hanno mai visto. ([`src/predict.py`](src/predict.py))
9. **Livelli di confidenza, non falsa certezza** — ogni risultato porta un
   livello di `confidence` e un flag `evidence_backed`. Un verdetto solo-modello
   su un claim breve fuori dominio (esattamente dove i modelli sbagliano con
   sicurezza) è presentato come **segnale di screening a bassa confidenza da
   verificare**, mai come verità acquisita. Questo abbassa solo la confidenza —
   non ribalta *mai* FAKE→REAL, quindi la garanzia zero-falsi-negativi resta.
10. **Rilevamento delle tecniche di manipolazione (prebunking / inoculazione)**
    — un livello robusto a dominio e tempo che segnala *come* un testo cerca di
    persuadere (appello alla conoscenza nascosta, fonte non verificabile,
    autorità fabbricata, linguaggio della paura, falsa certezza, urgenza,
    noi-contro-loro), ognuna con una breve spiegazione. Seguendo Roozenbeek &
    van der Linden (2019), nominare la tecnica inocula il lettore meglio di un
    timbro vero/falso. È ortogonale al classificatore — sul benchmark si accende
    su 17 delle 23 bufale in stile classico ma su nessuna delle affermazioni
    vere che i modelli sovra-segnalano — e alza il flag di revisione senza mai
    cambiare il label. Condivide però gran parte del punto cieco del
    classificatore sulle bufale fluenti in stile IA (2 su 6 reali segnalate,
    0 su 8 scritte a mano): vedi sotto. ([`src/manipulation.py`](src/manipulation.py))
11. **Explainability e feedback loop** — la SVM è lineare, quindi il suo
    punteggio si scompone esattamente in contributi per-token (TF-IDF ×
    coefficiente); la demo mostra le parole che spingono verso FAKE e verso
    REAL. Un form 👍/👎 + etichetta-corretta registra le correzioni in un file
    JSONL locale — la materia prima per un set di valutazione dal mondo reale e
    per un futuro riaddestramento sui casi difficili noti.
    ([`src/explain.py`](src/explain.py), [`src/feedback.py`](src/feedback.py))

Il benchmark fuori dominio resta offline e riproducibile (nessuna chiamata
live), così questi livelli migliorano il comportamento reale senza gonfiare
il 78,1% misurato. Un test di regressione
([`tests/test_benchmark_invariants.py`](tests/test_benchmark_invariants.py))
verifica la garanzia che regge davvero — **zero falsi negativi sulla
disinformazione in stile classico** — e una soglia minima di accuratezza
complessiva, senza fingere che il divario sullo stile IA-fluente qui sotto non
esista.

## La disinformazione generata dall'IA è più difficile da individuare

Questa è una risposta diretta e parziale alla domanda di ricerca dietro questo
progetto (vedi *Motivazione* sopra): **rimuovere gli indicatori stilistici
della disinformazione toglie anche a chi si difende la capacità di
riconoscerla?** La sezione documenta anche una correzione metodologica fatta a
metà progetto — il risultato è diventato più debole, e più affidabile, dopo
aver corretto un difetto in come era stato misurato la prima volta.

Il benchmark a 64 scenari etichetta ogni claim fabbricato con uno `style`:

- **`human_typical`** (23 scenari) — i tropi classici della disinformazione:
  *accordi segreti*, *un memo trapelato*, *secondo un whistleblower*, *fonti
  anonime*. È il registro di cui sono pieni i corpora di addestramento
  (ISOT/WELFake, in gran parte scritti da umani, pre-2021).
- **`ai_fluent`** (14 scenari) — lo stesso tipo di claim fabbricato, scritto
  come prosa fluente, calma, attribuita a una fonte, senza nessuno di quei
  tropi: un'istituzione dal nome plausibile, una statistica specifica, una
  cautela metodologica — il registro che un LLM moderno produce di default
  quando gli si chiede di scrivere testo persuasivo (vedi Goldstein et al.,
  2023; Helmus & Chandra, 2024, citati in *Motivazione*).

**Perché `ai_fluent` è diviso per `provenance`, e perché questo conta più del
numero in testa.** La prima versione di questo benchmark aveva tutti gli
scenari `ai_fluent` scritti a mano per questo progetto — da un LLM, che
conosceva esattamente cosa cerca il detector di tecniche di manipolazione di
questo stesso sistema. Questo è circolare: misura se una difesa può essere
elusa da qualcuno che già sa come funziona, non se la disinformazione
*effettivamente prodotta* da un LLM la elude. Per correggere questo, 6 dei 14
scenari `ai_fluent` (tutti quelli lunghi, in formato articolo) sono stati
sostituiti con vera disinformazione generata da ChatGPT-3.5 — riscritture di
bufale documentate e scritte da umani, pescate (seed casuale fisso, nessuna
selezione mirata) dal **dataset LLMFake** di Chen & Shu
([ICLR 2024](https://github.com/llm-misinformation/llm-misinformation), il
cui stesso risultato è che "la disinformazione generata da LLM può essere più
difficile da individuare per umani e detector rispetto alla disinformazione
scritta da umani con la stessa semantica"). Prima di usarlo, il suo
sotto-corpus CoAID è stato controllato e scartato: un controllo di
sovrapposizione testuale ha trovato che filtra nei nostri stessi dati di
training COVID-19, il che lo avrebbe reso un test di leakage, non un test
adversarial. Gli 8 scenari `ai_fluent` brevi restano scritti a mano — non
esiste un corpus pubblico pulito di disinformazione *breve* parafrasata da
LLM (CoAID sarebbe stato il candidato, ed è esattamente quello che è dovuto
essere escluso) — e restano nel benchmark come dato esplorativo dichiarato,
non come titolo.

Recall misurato (il "tasso di cattura" delle bufale),
`python -m src.evaluate --adversarial`:

| Stile / provenienza | n | Recall (tasso di cattura) | Layer di manipolazione segnala |
|---|---|---|---|
| `human_typical` (scritto a mano) | 23 | **100%** (0 mancate) | 17/23 (74%) |
| `ai_fluent` / **`external_dataset`** (reale, non scritto da questo progetto) | 6 | **83,3%** (1 mancata) | 2/6 (33%) |
| `ai_fluent` / `hand_authored` (esplorativo, con la riserva di circolarità sopra) | 8 | 50,0% (4 mancate) | 0/8 (0%) |

**Il risultato citabile è la riga centrale.** Ristretto agli stessi due domini
(politica, intrattenimento/misto) così il confronto è alla pari,
`human_typical` ottiene 100% (13/13) contro `ai_fluent`/`external_dataset` a
83,3% (5/6) — un divario reale, non circolare, di circa 17 punti. La riga in
basso è un'evidenza più debole: direzionalmente coerente, ma la costruzione a
mano non può escludere di essere stata implicitamente calibrata contro la
logica di rilevamento di questo stesso sistema.

**Una ritrattazione, nell'interesse della stessa onestà che questo README
chiede al sistema.** La prima versione di questa sezione affermava anche una
"seconda conferma indipendente" dell'effetto tramite la lunghezza
dell'input: l'accuratezza complessiva sugli scenari lunghi, in formato
articolo, misurava 56,2%, contro 75,0% sui claim brevi, attribuita al fatto
che la fluenza generativa non degrada con la lunghezza come invece fa la
coerenza di un testo scritto da un umano. Quell'affermazione non sopravvive
alla correzione qui sopra: con 6 degli scenari lunghi scritti a mano sostituiti
da dati reali esterni, **l'accuratezza sugli scenari lunghi è ora 87,5%** —
più alta di quella sui brevi (75,0%), l'opposto dell'affermazione originale.
Il presunto "effetto lunghezza" non era affatto una proprietà della lunghezza:
era un artefatto del fatto che ogni scenario lungo, in quella versione, era
stato scritto a mano in modo avversariale. Viene rimosso qui invece che
silenziosamente aggiornato, perché l'errore — fidarsi di un effetto misurato
interamente su testo auto-scritto — è esso stesso la lezione: **qualunque
affermazione su cosa elude questo sistema, se misurata solo su testo scritto
dallo stesso sistema (o dal suo stesso sviluppatore), ha bisogno di dati
indipendenti e non circolari prima di essere considerata attendibile.**

**Cosa significa, e cosa non significa, per le garanzie del sistema.** La
garanzia di zero falsi negativi ripetuta in tutto questo README è reale, ma
ora è esplicitamente delimitata: regge per la disinformazione scritta nel modo
in cui è stata storicamente scritta la disinformazione documentata. Contro la
fabbricazione fluente parafrasata da IA regge meno bene — un divario reale ma
moderato (100% contro 83,3% su dati sorgente indipendenti), non il punto
cieco quasi totale suggerito da una misurazione precedente,
metodologicamente circolare. Coerentemente con la sezione *Motivazione*
sopra, questo non è un fallimento del design del sistema ma la sua conferma:
uno **strumento di screening dentro un processo umano**, non un arbitro
automatico. Il passo onesto successivo — non ancora implementato — sarebbe
espandere il campione `external_dataset` (es. altri metodi di generazione
dello stesso dataset LLMFake, o le sue varianti Llama2/Vicuna) per ottenere un
campione più grande e ancora indipendente, in particolare per claim brevi.

## Pipeline e figure

La pipeline completa è documentata in [PIPELINE.md](PIPELINE.md). Mostra il
flusso end-to-end dai dataset grezzi al deploy su Streamlit.

Il livello di reporting è riassunto in [reports/README.md](reports/README.md),
che spiega cosa dimostra ciascun grafico qui sopra e perché è rilevante per il
sistema finale. Nel complesso, le tre figure documentano i modi di fallimento
che hanno spinto il progetto finale ad allontanarsi da un benchmark guidato
dalla sola accuratezza, verso un workflow di retrieval e revisione.

## Riepilogo della pipeline

```mermaid
flowchart LR
    A[Dataset grezzi] --> B[Pulizia / deduplicazione / filtri]
    B --> C[Split train-test]
    C --> D[TF-IDF + tokenizer]
    D --> E[SVM + Bi-GRU/Bi-LSTM -> TFLite]
    E --> F[Retrieval semantico sul corpus di riferimento]
    F --> G[Ensemble + flag di revisione]
    G --> H[Demo Streamlit]
```

## Risultati (tutti misurati, tutti riproducibili)

**In-domain** — test set condiviso, `python -m src.train` →
[`models/metrics.json`](models/metrics.json):

| Modello | Accuratezza | Precisione (fake) | Recall (fake) | F1 (fake) |
|---|---|---|---|---|
| SVM (TF-IDF, calibrata) | 95,3% | 94,8% | 94,9% | 94,8% |
| Bi-GRU | 92,9% | 93,0% | 91,0% | 92,0% |
| Bi-LSTM | 92,9% | 94,1% | 89,9% | 92,0% |
| **Ensemble (media)** | **94,6%** | 94,5% | 93,3% | 93,9% |

**Fuori dominio** — 64 scenari adversarial (hoax plausibili, verità scomode —
claim brevi e articoli lunghi, sei domini, due stili di disinformazione),
`python -m src.evaluate --adversarial` →
[`benchmarks/adversarial_results.json`](benchmarks/adversarial_results.json):

| Dominio | Accuratezza | Falsi positivi | Falsi negativi | Segnalati per revisione |
|---|---|---|---|---|
| Politica | 81,2% | 3 | 0 | 5 |
| COVID | 84,6% | 1 | 1 | 4 |
| Misto | 75,0% | 3 | 1 | 7 |
| Economia | 71,4% | 1 | 1 | 3 |
| Scienza | 66,7% | 0 | 2 | 2 |
| Tecnologia | 83,3% | 1 | 0 | 2 |
| **Totale** | **78,1%** | 9 | 5 | 23 |

Per **lunghezza** — claim brevi vs. scenari lunghi in formato articolo:

| Lunghezza | n | Accuratezza |
|---|---|---|
| Breve | 48 | 75,0% |
| Lunga | 16 | 87,5% |

*(Una versione precedente di questa tabella mostrava gli scenari lunghi con
un'accuratezza molto peggiore (56,2%) e la presentava come una seconda
conferma indipendente dell'effetto della fluenza IA. Non lo era: ogni scenario
lungo in quella versione era scritto a mano, e "l'effetto lunghezza" è
scomparso — anzi si è ribaltato — una volta sostituiti 6 di quegli scenari
con testo reale, esterno. Vedi la ritrattazione nella sezione *"La
disinformazione generata dall'IA è più difficile da individuare"* sopra.)*

Per **stile**, solo scenari FAKE — il recall (tasso di cattura delle bufale)
che conta di più per uno strumento anti-disinformazione:

| Stile | n | Recall (tasso di cattura) |
|---|---|---|
| `human_typical` (tropi classici) | 23 | **100,0%** |
| `ai_fluent` (stile IA, fluente — media aggregata, vedi sopra) | 14 | 64,3% |

`ai_fluent` mescola due provenienze con un peso probatorio molto diverso —
vedi *"La disinformazione generata dall'IA è più difficile da individuare"*
sopra per la scomposizione citabile e non circolare (83,3% sui dati reali
esterni contro 50,0% sul testo scritto a mano).

I falsi positivi ripetono il risultato originale — un **falso positivo su
un'affermazione vera** ("Donald Trump ha vinto le elezioni del 2016…" →
FAKE): la finestra di addestramento 2015–2017 ha insegnato ai classificatori
che brevi affermazioni fattuali sulla politica USA *assomigliano* a esche da
fake news, e il layer di retrieval deliberatamente non le "salva" più
trattando un articolo reale sullo stesso tema come una prova (vedi la sezione
successiva). Ogni falso negativo è una bufala in stile `ai_fluent` — la
garanzia zero-falsi-negativi regge **solo** per la disinformazione in stile
classico, non per la fabbricazione fluente in stile IA (vedi sopra per
quantificare esattamente quanto non regge).

## Cosa dicono i grafici

I grafici del report rispondono a tre domande prima ancora di guardare
l'accuratezza:

1. Il dataset lascia trapelare l'etichetta attraverso lo stile?
2. L'etichetta trapela attraverso marcatori di fonte?
3. La finestra temporale è troppo stretta per generalizzare?

Se anche solo una di queste risposte è "sì", le metriche del modello vanno
lette come stime valide solo in-domain. Per questo il portfolio mette in
primo piano il benchmark adversarial e la pipeline di retrieval/revisione,
invece del solo numero di accuratezza.

## Due usi molto diversi degli embeddings

TF-IDF, una SVM lineare e due piccole RNN sembrano datati rispetto ai
classificatori testuali attuali — perciò entrambi gli usi possibili degli
embeddings transformer sono stati testati su questo progetto, con risultati
opposti e ugualmente istruttivi.

**Classificazione: testata, respinta.** `experiments/` sostituisce la
baseline TF-IDF con embeddings di frase
([`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
più un classificatore lineare calibrato, addestrato e valutato sullo
*stesso identico* dataset fuso e split di `src.train`
(`experiments/embeddings_baseline.py`, `experiments/embeddings_adversarial.py`).

| | In-domain | Fuori dominio (benchmark a 30 scenari) |
|---|---|---|
| Ensemble attuale (TF-IDF + SVM/GRU/LSTM) | 94,6% | 80,0% |
| Embeddings MiniLM + classificatore lineare | 88,5% | 60% |

*Fotografia storica, congelata per comparabilità: misurata sul benchmark
originale a 30 scenari, prima che fosse espanso a 64 (aggiungendo nuovi
domini, articoli lunghi e la suddivisione di stile human-typical/ai-fluent —
vedi "La disinformazione generata dall'IA è più difficile da individuare"
sopra). Questa ablation era un confronto una-tantum tra approcci di
classificazione, non un benchmark mantenuto, quindi non è stata rieseguita
sul set espanso.*

Il classificatore basato su embeddings ha perso su entrambi i fronti — il
divario più netto è su WELFake (67,1% contro 86,9%) e sul dominio
adversarial "misto". È la conseguenza misurata del leakage documentato in
[`notebooks/01_dataset_bias_analysis.ipynb`](notebooks/01_dataset_bias_analysis.ipynb):
la distinzione fake/vero in questi corpora è guidata in gran parte da stile
superficiale e marcatori di fonte (punteggiatura, maiuscole, la dicitura
`(Reuters)`), e TF-IDF è costruito apposta per sfruttare esattamente quel
segnale letterale. Un modello di embedding semantico è costruito per
esserne invariante — quindi su questo dataset, capire *meglio* il
significato è uno svantaggio per la classificazione.

**Retrieval: testato, adottato.** Trovare il claim *noto* più vicino è un
compito diverso dalla classificazione, ed è esattamente ciò per cui gli
embeddings semantici sono fatti: far corrispondere "il vaccino COVID altera
il tuo codice genetico" a un claim salvato sul vaccino che "altera
permanentemente il DNA", pur con un vocabolario condiviso quasi nullo —
qualcosa che il vecchio livello di riferimento TF-IDF, basato sulla
sovrapposizione letterale di termini, non poteva fare per costruzione.
`src/rag.py` ora calcola una volta sola gli embeddings dei ~68k snippet del
corpus di riferimento (`REF_EMBEDDINGS_FILE`, committato, ~46 MB) e
confronta le query per similarità coseno. Anche i pesi del modello stesso
(`models/embedding_model/`, ~88 MB) sono committati invece di essere scaricati
da Hugging Face Hub a runtime — i container di Streamlit Cloud ripartono da un
filesystem pulito a ogni redeploy, e questo livello gira su *ogni* previsione,
non solo sul retrieval live, quindi un download da Hub all'avvio a freddo era
un rischio concreto: se la rete ha un intoppo, l'app semplicemente non parte.

**Il segnale di retrieval è volutamente asimmetrico — e l'asimmetria conta
più del numero in cima.** Una versione iniziale lasciava che *qualsiasi*
corrispondenza, vera o falsa, influenzasse il verdetto. Otteneva un 83,3%
adversarial più alto — ma in parte fabbricando verità: un'affermazione falsa
condivide di continuo il proprio argomento con notizie reali ("il vaccino
altera il tuo DNA" sta proprio accanto ad articoli veri sulla genetica del
COVID), quindi la demo mostrava un pannello verde "REAL / SUPPORTED"
*direttamente sotto un verdetto FAKE rosso*, arrivando a dare il via libera a
una teoria complottista sul vaccino. È esattamente il segnale sbagliato per
uno strumento anti-disinformazione. Perciò il layer di riferimento ora è
asimmetrico:

- Corrispondere a un claim **falso** noto è vera evidenza di falsità — alza il
  punteggio già da una similarità modesta, e un match quasi letterale può
  scavalcare l'ensemble.
- Corrispondere a un articolo **vero** noto afferma REAL solo se è *quasi
  letterale* (`REF_OVERRIDE_THRESHOLD = 0,90`); la semplice vicinanza tematica
  è mostrata come evidenza neutra ("lo snippet noto più vicino è reale al
  69%"), mai come un verdetto.

Questo costò all'epoca circa tre punti di accuratezza adversarial (83,3% →
80,0%, misurati sul benchmark originale a 30 scenari; un'affermazione COVID
vera non più "salvata" da un articolo reale sullo stesso tema) — un costo che
vale la pena pagare: i pannelli non possono più contraddire il verdetto, e la
demo non presenta mai un'affermazione falsa come supportata. La similarità di
embeddings semplicemente non separa "stesso claim, riformulato" da "stesso
argomento, claim diverso" con abbastanza nettezza da
essere trattata come segnale di verità, ma solo come evidenza recuperata.

**Perché entrambi sono diventati sostenibili insieme:** le RNN ora girano
come modelli TFLite tramite l'interprete `ai-edge-litert` (~10 MB) invece
del runtime TensorFlow completo (~500+ MB solo per il framework, a
prescindere dalla dimensione del modello). La memoria di picco misurata per
l'intero sistema — SVM, entrambe le RNN, il corpus di riferimento e il
modello di embeddings insieme — è di **~600 MB**, contro un limite di 1 GB
del piano gratuito di Streamlit Cloud. Tenere TensorFlow e PyTorch insieme
non ci sarebbe stato; rinunciare alle RNN o agli embeddings sarebbe stato un
compromesso inutile. L'addestramento avviene ancora con TensorFlow completo
(`requirements-train.txt`); solo l'app deployata doveva cambiare.

## Collocazione nella tassonomia del disordine informativo

"Fake news" è un'etichetta scientificamente inadeguata: il framework
*Information Disorder* di Wardle & Derakhshan (Consiglio d'Europa, 2017)
distingue **misinformation** (falso, condiviso senza intento dannoso),
**disinformation** (falso, intenzionalmente dannoso) e **malinformation**
(contenuto genuino usato per nuocere). Un classificatore testuale può
occuparsi solo del *segnale di falsità del contenuto* delle prime due — è
cieco all'intento, e per costruzione alla malinformation, dove il contenuto è
vero. Questa è una seconda ragione strutturale (oltre all'accuratezza
misurata fuori dominio) per cui il sistema è inquadrato come un **aiuto allo
screening dentro un processo umano**, non un arbitro automatico della verità.

Il benchmark adversarial versionato segue la stessa logica che la
letteratura sulla sicurezza cognitiva applica alle istituzioni — *non puoi
difendere ciò che non hai testato*: i 64 scenari restano nel repository come
uno stress test permanente e ripetibile, non un esperimento occasionale.

## Struttura del repository

```
├── app.py                  Demo Streamlit (solo UI)
├── src/
│   ├── config.py           ogni percorso, iperparametro e soglia
│   ├── data.py             caricamento / filtri / fusione / protocollo di split unificati
│   ├── train.py            addestra SVM + GRU + LSTM, esporta TFLite, scrive metrics.json
│   ├── predict.py          ScreeningSystem: ensemble + euristica + flag di revisione
│   ├── evaluate.py         report in-domain e benchmark adversarial
│   ├── rag.py              retrieval sul corpus di riferimento (embeddings semantici)
│   ├── claim_rag.py        analisi di retrieval per singolo claim
│   ├── external_retrieval.py  evidenza live (Google Fact Check / Wikipedia / GDELT)
│   ├── manipulation.py     livello prebunking: tecniche di manipolazione retorica
│   ├── explain.py          contributi per-token della SVM (spiegazione lineare esatta)
│   ├── feedback.py         log append-only delle correzioni utente (JSONL)
│   └── tokenizer.py        tokenizer indipendente dal framework (niente TF in produzione)
├── tests/                  suite pytest: protocollo di split, logica ensemble, retrieval
├── models/                 artefatti addestrati incl. RNN TFLite (~8 MB) e il
│                           modello di embedding committato (~88 MB)
├── reference_corpus/       snippet noti veri/falsi + embeddings (~55 MB)
├── benchmarks/             scenari versionati + risultati misurati
├── experiments/            alternative testate e scartate (vedi sopra)
├── notebooks/              analisi del bias del dataset (il "perché" del design)
├── reports/figures/        grafici esportati
└── data/                   dataset (non committati — vedi data/README.md)
```

## Avvio rapido

```bash
# Python 3.10 o 3.11
pip install -r requirements.txt

# Avvia la demo con i modelli già committati
streamlit run app.py

# Riproduci tutto da zero — servono i dataset (vedi data/README.md)
# E TensorFlow, usato solo per l'addestramento; l'app in sé non ne ha bisogno:
pip install -r requirements-train.txt
python -m src.train                  # ~10 min su CPU
python -m src.evaluate               # tabella metriche in-domain
python -m src.evaluate --adversarial # benchmark fuori dominio

# Esegui la suite di test (protocollo di split, logica ensemble, retrieval)
pip install -r requirements-dev.txt
python -m pytest tests/
```

## Deploy su Streamlit Cloud

Questo repository è già configurato per un deploy standard su Streamlit
Cloud.

Puoi aprire l'app già deployata direttamente su
https://fake-news-screening.streamlit.app/.

1. Collega il repository GitHub `lauratonsi/Fake_News_Screening`.
2. Usa `app.py` come entry point.
3. Mantieni `main` come branch predefinito.
4. Lascia che Streamlit installi le dipendenze da `requirements.txt` (include
   un indice PyTorch CPU-only per `torch`, così non scarica una build CUDA
   da svariati GB).
5. In **Advanced settings**, imposta la versione di Python a **3.11**.
6. I default di tema/server sono impostati in `.streamlit/config.toml`.

Se il deploy va a buon fine, la demo dovrebbe caricare i modelli già
committati in `models/` e `reference_corpus/` e funzionare senza bisogno di
riaddestramento né di TensorFlow — vedi *"Perché entrambi sono diventati
sostenibili insieme"* sopra per il conto della memoria dietro questa scelta.

## Retrieval live: configurazione e aspettative oneste

Il livello live (`src/external_retrieval.py`) interroga fonti gratuite per
ogni claim, in ordine di precedenza:

1. **Google Fact Check Tools** — solo se è impostata
   `GOOGLE_FACTCHECK_API_KEY`. L'unica fonte che restituisce un vero
   *verdetto* di fact-checking, quindi vince.
2. **Wikipedia** (API di ricerca MediaWiki, senza chiave) — *contesto*
   tematico affidabile e veloce. È il default su cui si può contare, quello che
   fa sì che il pannello "Retrieval live" mostri davvero evidenza concreta. È
   contesto, mai un verdetto.
3. **GDELT** (senza chiave) — una ricerca notizie *live* di ultima istanza. Il
   suo endpoint gratuito condiviso è pesantemente rate-limited (HTTP 429) e
   inaffidabile, quindi entra in gioco solo quando le due sopra non
   restituiscono nulla; è tenuto per completezza, non è una fonte su cui contare.

Wikipedia restituisce qualcosa in tema praticamente per ogni claim, attuale o
storico — è il motivo per cui ha sostituito GDELT come default: un claim sulla
Federal Reserve restituisce l'articolo *Federal Reserve*, un claim sull'elezione
del 2016 restituisce *Donald Trump* / *Hillary Clinton 2016 presidential
campaign*. È contesto da leggere, non un verdetto di verità — solo il percorso
Google Fact Check ne afferma uno.

**Per attivare il percorso di qualità superiore di Google Fact Check:**
1. Nella Google Cloud Console, abilita la "Fact Check Tools API" e crea una
   chiave API.
2. In locale: `export GOOGLE_FACTCHECK_API_KEY=la-tua-chiave` prima di
   `streamlit run app.py`.
3. Su Streamlit Community Cloud: apri **Settings → Secrets** dell'app e
   aggiungi
   ```toml
   GOOGLE_FACTCHECK_API_KEY = "la-tua-chiave"
   ```
   Streamlit Cloud espone i Secrets all'app come variabili d'ambiente, quindi
   non serve alcuna modifica al codice.

Senza chiave, l'app funziona esattamente come documentato sopra — Google
Fact Check viene saltato e Wikipedia è la fonte live di default.

## Limitazioni oneste

- Solo inglese; i corpora di addestramento si fermano sostanzialmente al
  2020 — gli eventi attuali sono fuori dominio.
- Il lookup di riferimento riconosce claim *già noti* (ora anche
  riformulati — vedi sopra); non può verificarne di genuinamente nuovi. La
  sua ricerca top-1 per vicinanza può anche confondere "stesso argomento"
  con "stesso claim" su input ambigui, motivo per cui scavalcare del tutto
  l'ensemble è riservato ai match quasi letterali
  (`REF_OVERRIDE_THRESHOLD = 0,90`).
- Le RNN sono addestrate su un sottocampione di 5.000 articoli (budget CPU);
  la SVM vede l'intero training set.
- L'accuratezza fuori dominio (78,1% complessivo) è il numero che conta per
  un uso reale, ed è il motivo per cui qualunque deploy di un sistema come
  questo richiede un essere umano nel ciclo.
- **La disinformazione fluente in stile IA è un punto debole misurato e
  moderato, non teorico.** Il recall sulle bufale in stile classico è 100%;
  su vere riscritture ChatGPT-3.5 di disinformazione documentata, sorgente
  indipendente, scende all'83,3% (vedi *"La disinformazione generata dall'IA
  è più difficile da individuare"* sopra). Sia il classificatore sia il layer
  delle tecniche di manipolazione si agganciano in parte a caratteristiche di
  superficie di *come gli umani hanno storicamente scritto* la
  disinformazione, e nessuno dei due sostituisce del tutto quel segnale
  quando è assente — ma il divario è reale, non il punto cieco quasi totale
  che una misurazione precedente, metodologicamente circolare, in questo
  stesso repository aveva suggerito.
