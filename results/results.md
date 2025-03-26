# Risultati e Analisi delle Performance
### Indice
1. [Ant-v5](#1-ant-v5)  
    1.1 [Interpretazione dei Risultati - Ant-v5](#11-interpretazione-dei-risultati---ant-v5)  
2. [Hopper-v5](#2-hopper-v5)  
    2.1 [Interpretazione dei Risultati](#21-interpretazione-dei-risultati)  
3. [Humanoid-v5](#3-humanoid-v5)  
    3.1 [Interpretazione dei Risultati](#31-interpretazione-dei-risultati)  
### 1. **Ant-v5**  
Osservando il file `results\evaluation\metrics_Ant-v5.json` è possibile visualizzare le seguenti metriche ottenute riguardo la ricompensa media negli episodi di valutazione (mean) e la deviazione standard (std):
* `TD3`: "mean": 0.04, "std": 0.05;
* `SAC`: "mean": -0.04, "std": 0.17;
* `PPO`: "mean": 0.09, "std": 0.09;
* `A2C`: "mean": 1.27,  "std": 0.21;
* `Random`: "mean": -45.02, "std": 41.97;
* `DQN`: "mean": 401.73, "std": 403.74;

### 1.1 **Interpretazione dei Risultati - Ant-v5**  
Per l'analisi dei risultati bisogna tenere conto che tutti i modelli sono stati addestrati in ambieni normalizzati pertanto le reward migliori sono quelle tanto più vicine al valore zero. Possiamo quindi dedurre che gli *algoritmi ottimali* (Top Performers) sono **TD3 e SAC** che dominano con valori assoluti medi circa **0.04**, entrambi vicinissimi allo zero. **TD3** è leggermente superiore avendo una deviazione standard **5 volte inferiore** a SAC (0.05 vs 0.17), indicando maggiore **consistenza**. Mentre **SAC** mostra una lieve polarizzazione negativa (-0.04), ma trascurabile nel contesto. **PPO**, con media: 0.09, è accettabile, ma più distante dallo zero rispetto a TD3 o SAC.

Per questo agente risultano essere algoritmi fuori target **A2C, DQN e la policy random**. **A2C** (1.27) e **DQN** (401.73) hanno infatti delle medie **fuori scala**, segnalando per A2C inadeguatezza verso l'agente o un numero di time stamp insufficiente, mentre per DQN incompatibilità radicale con gli ambienti normalizzati (valori non controllati) anche in virtù della discretizzazione richeista. Il modello che segue una **policy random** (-45.02) funge da baseline negativa, come atteso. 

Analizzando la stabilità **TD3** è il più stabile (std: 0.05), seguito da SAC (0.17) e PPO (0.09). Mentre **DQN** mostra varianza catastrofica (std: 403.74), rendendolo inutilizzabile.

Di seguito sono riportati i grafici per confronto di quanto appena osservato.
<div style="display: flex; justify-content: space-between;">
    <img src="./plots/Comparison of Algorithms rewards during 10 episodes for Ant-v5.png" alt="Ant-v5 Rewards" width="45%">
    <img src="./plots/Rewards during episodes for Ant-v5.png" alt="Ant-v5 rewards during episodes" width="45%">
</div>

### 2. **Hopper-v5**
Le metriche ottenute con l'agente `Hopper-v5` e conservate nel file `results\evaluation\metrics_Hopper-v5.json` mostrano i seguenti valori: 
* `TD3`: "mean": 1.12, "std": 1.01;
* `SAC`: "mean": 0.32, "std": 0.27;
* `PPO`: "mean": 0.24, "std": 0.26;
* `A2C`: "mean": 0.15,  "std": 0.18;
* `Random`: "mean": 22.18, "std": 16.77;
* `DQN`: "mean": 435.96, "std": 156.88;

### 2.1 **Interpretazione dei Risultati** 
Analizzando questi valori il modello **A2C** risulta essere quello con miglior performance relativa poiché presenta una media di **0.15**. Dal punto di vista della stabilità, presenta una deviazione standard di **0.18** che è accettabile per ambienti normalizzati. Tuttavia, alcuni episodi raggiungono reward di **0.41**, indicando picchi non controllati. I modelli **PPO e SAC** sono subottimali ma con potenziale:  
   - **PPO** (media: 0.24) mostra ricompense più consistenti (min: **0.007**), ma massimi elevati (**0.72**).  
   - **SAC** (media: 0.32) ha picchi preoccupanti (**0.78**) e minimi meno vicini allo zero (**0.055**).  
Per questo ambiente il modello **TD3** fornisce performance insoddisfacenti, infatti le metriche mostrano una media fuori scala (**1.12**) e una varianza elevata (**1.01**), segnale di un insufficiente addestramento. Infine, **DQN e policy random** sono totalmente inaffidabili, confermando incompatibilità con ambienti discretizzati e normalizzati per DQN (media: 435.96), e l'inefficacia di una policy non addestrata per questo task (media: 22.18).  

Poiché TD3 per il task precedente (Ant-v5) mostra buoni risultati mentre scarsi per questo, potrebbe necessitare di una riprogettazione degli iperparametri, come ad esempio la riduzione del fattore di sconto per evitare l'accumulo di ricompense. Un **problema comune** a tutti i modelli, suggerito dai dati, è che **nessun algoritmo** è correttamente allineato al range [-1, 1]. Una pipeline di **reward shaping** potrebbe essere necessaria per questo ambiente.  

Di seguito sono riportati i grafici per confronto di quanto appena osservato.
<div style="display: flex; justify-content: space-between;">
    <img src="./plots/Comparison of Algorithms rewards during 10 episodes for Hopper-v5.png" alt="Hopper-v5 Rewards" width="45%">
    <img src="./plots/Rewards during episodes for Hopper-v5.png" alt="Hopper-v5 rewards during episodes" width="45%">
</div>

### 3. **Humanoid-v5**
Dal file delle metriche `results\evaluation\metrics_Humanoid-v5.json` osserviamo che:
* `TD3`: "mean": 0.0057, "std": 0.0041;
* `SAC`: "mean": 0.0050, "std": 0.0030;
* `PPO`: "mean": 0.0799, "std": 0.0714;
* `A2C`: "mean": 0.16,  "std": 0.13;
* `Random`: "mean": 121.19, "std": 37.71;

### 3.1 **Interpretazione dei Risultati**
Da questi dati si deduce che **SAC e TD3** sono dei modelli eccellenti. Infatti, **SAC** (media: **0.0050**) e **TD3** (media: **0.0057**) hanno valori medi **estremamente vicini allo zero**, con deviazioni standard minime (**0.0030** e **0.0041**). **SAC** è leggermente superiore grazie a una varianza inferiore (**8.93e-6** vs 1.66e-5 di TD3). Entrambi mostrano ricompense **strettamente controllate** durante tutti gli episodi di test (max: 0.010 per SAC, 0.015 per TD3). **PPO** (media: 0.0799) ha una distribuzione di ricompense **instabile** (std: 0.0714), con picchi fino a **0.1976**, mentre **A2C** (media: 0.1582) è il **peggiore tra gli algoritmi addestrati**, con ricompense disperse (max: **0.4203**, min: **0.013**). Per un agente così complesso la **policy random** è totalmente inutilizzabile. Infatti presenta una media fuori scala (**121.19**) e una varianza elevata (**37.71**), tipica di un agente non addestrato. Di seguito è mostrata una tabella riasuntiva delle performance in un'ottica di stabilità.

| Modello   | Deviazione Standard | Interpretazione                     |  
|-----------|---------------------|-------------------------------------|  
| **SAC**   | 0.0030              | Stabilità ottimale                  |  
| **TD3**   | 0.0041              | Leggermente più rumoroso di SAC     |  
| **PPO**   | 0.0714              | Instabilità critica                 |  
| **A2C**   | 0.1334              | **Inaccettabile** per ambienti complessi |  

Di seguito sono riportati i grafici per confronto di quanto appena osservato.
<div style="display: flex; justify-content: space-between;">
    <img src="./plots/Comparison of Algorithms rewards during 10 episodes for Humanoid-v5.png" alt="Humanoid-v5 Rewards" width="45%">
    <img src="./plots/Rewards during episodes for Humanoid-v5.png" alt="Humanoid-v5 rewards during episodes" width="45%">
</div>