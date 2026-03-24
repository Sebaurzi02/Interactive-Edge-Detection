
# Interactive Edge Detection Framework

Questo repository contiene lo sviluppo di un **framework interattivo per l’edge detection**, progettato per confrontare e analizzare **metodi classici** e **modelli basati su deep learning**.

Il progetto è focalizzato su:

* confronto visivo tra algoritmi
* analisi qualitativa dei risultati
* sperimentazione interattiva
* interfaccia grafica intuitiva

Include sia un workflow in **Jupyter Notebook** che una **UI standalone in Python (Tkinter + Matplotlib)**.

---

# Obiettivi del progetto

Il framework consente di:

* Caricare ed elaborare:

  * immagini singole
  * dataset standard (es. **BSDS500**)
* Selezionare dinamicamente l’algoritmo di edge detection
* Visualizzare **input vs output**
* Analizzare il comportamento degli algoritmi al variare dei parametri
* Confrontare approcci classici e deep learning in modo intuitivo

---

#  Metodi supportati

##  Edge Detector Classici

* **Canny (implementazione custom)**

##  Edge Detector Deep Learning

* **TEED (Tiny and Efficient Edge Detector)**
* **DexiNed (Dense Extreme Inception Network)**

---

#  Canny Edge Detector (Implementazione Custom)

L’algoritmo di **Canny** è implementato **completamente da zero**, senza l’uso di librerie esterne, seguendo la formulazione teorica classica.

### Pipeline:

1. Gaussian smoothing
2. Calcolo del gradiente
3. Non-Maximum Suppression
4. Double threshold
5. Edge tracking by hysteresis

### Vantaggi:

* Controllo completo dell’algoritmo
* Possibilità di analisi didattica
* Confronto diretto con modelli deep learning

---

#  Modelli Deep Learning

##  TEED – Tiny and Efficient Edge Detector

TEED è una **CNN leggera** progettata per bilanciare:

* accuratezza
* efficienza computazionale

### Caratteristiche:

* Architettura compatta
* Feature multi-scala
* Robustezza a rumore e texture
* Nessun tuning manuale dei parametri

### Dataset:

* Training: **BIPED**
* Test: **BSDS500**

---

##  DexiNed – Dense Extreme Inception Network

DexiNed è un modello più profondo e complesso, progettato per catturare:

* dettagli fini
* strutture semantiche

### Architettura:

* Dense blocks
* Moduli Inception
* Side outputs multi-scala
* Fusione finale delle edge maps

### Caratteristiche:

* Edge detection multi-scala
* Buona generalizzazione
* Adatto a scene complesse

### Dataset:

* Training: **BIPED**

---

#  Interfaccia Grafica (UI)

Il progetto include una **UI standalone avanzata** sviluppata in:

* **Tkinter**
* **Matplotlib**

---

##  Funzionalità principali

###  Controllo algoritmi

* Selezione dinamica:

  * Canny
  * TEED
  * DexiNed

---

###  Parametri Canny interattivi

* Slider in tempo reale per:

  * threshold basso
  * threshold alto
  * sigma
  * hysteresis

---

###  Visualizzazione

* Input e Output affiancati
* Modalità:

  * edge map
  * overlay su immagine originale
* Navigazione tra immagini (dataset)

---

###  Tempo di esecuzione

* Visualizzazione del tempo per ogni algoritmo
* Utile per confronti qualitativi delle prestazioni

---

### Log e monitoraggio

* Output dettagliato delle operazioni
* Debug semplice degli errori

---

###  Esecuzione modelli

* TEED e DexiNed eseguiti tramite:

  * `subprocess`
  * thread separati

---

###  Progress Bar

* Feedback visivo durante l’esecuzione

---

###  Salvataggio risultati

* Export delle immagini generate
* Supporto PNG / JPG

---

###  Pulizia automatica

Alla chiusura del programma:

* eliminazione dati temporanei TEED
* eliminazione risultati DexiNed
* pulizia directory output

---

#  Dataset utilizzati

##  BSDS500

* 500 immagini naturali
* Annotazioni multiple
* Standard per test qualitativi

📎 [https://www.kaggle.com/datasets/balraj98/bsds500](https://www.kaggle.com/datasets/balraj98/bsds500)

---

##  BIPED

* Dataset per edge detection percettiva
* Scene urbane ad alta qualità
* Annotazioni accurate

📎 [https://www.kaggle.com/datasets/xavysp/biped](https://www.kaggle.com/datasets/xavysp/biped)

---

#  Modalità di utilizzo

##  Jupyter Notebook

* Analisi passo-passo

##  Applicazione Desktop

* UI completa
* Test su dataset
* Navigazione risultati
* Confronto visivo immediato

---

#  Note

* I modelli deep learning utilizzano:

  * implementazioni originali
  * pesi pre-addestrati
* Il progetto è focalizzato su:

  * studio
  * sperimentazione
  * presentazioni accademiche

---

> Questo progetto è stato sviluppato come ambiente interattivo per l’analisi e il confronto tra algoritmi di edge detection classici e basati su deep learning.

