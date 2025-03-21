### Gymnasium RL Agent Trainer - A Modular Framework for Algorithm Benchmarking, Hyperparameter Optimization, and Performance Evaluation in MuJoCo Environments

To read the README.md in English, click [here](README_english.md).

## Indice  
1. [Descrizione del Progetto](#1-descrizione-del-progetto)  
2. [Setup dell'Ambiente](#2-setup-dellambiente)  
3. [Struttura della Repository](#3-struttura-della-repository)  
4. [Agenti Supportati](#4-agenti-supportati)  
5. [Algoritmi Implementati](#5-algoritmi-implementati)  
6. [Utilizzo del Progetto](#6-utilizzo-del-progetto)  
    - 6.1 [Configurazione dei Parametri](#61-configurazione-dei-parametri)  
    - 6.2 [Esecuzione dell'ottimizzazione degli iperparametri](#62-esecuzione-dellottimizzazione-degli-iperparametri)  
    - 6.3 [Esecuzione dell'Addestramento](#63-esecuzione-delladdestramento)  
    - 6.4 [Valutazione degli Agenti e Presentazione dei Grafici](#64-valutazione-degli-agenti-e-presentazione-dei-grafici)  
    - 6.5 [Utilizzo di Modelli Pre-Addestrati](#65-utilizzo-di-modelli-pre-addestrati)

### 1. Descrizione del Progetto
Questo progetto è dedicato all'implementazione e alla valutazione di diversi algoritmi avanzati di Reinforcement Learning (RL) per l'addestramento di agenti autonomi in una varietà di ambienti Gymnasium, quali **HalfCheetah-v5**, **Ant-v5**, **Humanoid-v5** e **Hopper-v5**. L'obiettivo principale è fornire una piattaforma flessibile per sperimentare con molteplici algoritmi di RL, confrontare le loro prestazioni e sfruttare modelli pre-addestrati per un avvio rapido.

Il progetto supporta i seguenti algoritmi di RL: **Proximal Policy Optimization (PPO)**, **Soft Actor-Critic (SAC)**, **Actor-Critic (A2C)**, **Deep Q-Network (DQN)**, **Twin Delayed Deep Deterministic Policy Gradient (TD3)** e una **policy casuale** come baseline.

Una caratteristica fondamentale di questo progetto è la possibilità per l'utente di **specificare vari parametri**, tra cui i percorsi delle cartelle per il salvataggio dei modelli e dei risultati, se eseguire o meno l'ottimizzazione degli iperparametri e se effettuare una valutazione degli agenti addestrati, misurandone le performance e confrontandoli mediante grafici. Inoltre, il progetto fornisce **modelli pre-addestrati** per alcuni algoritmi e ambienti, consentendo agli utenti di testare rapidamente le prestazioni senza dover eseguire l'intero processo di ottimizzazione degli iperparametri e l'addestramento. Infine, vengono generati **grafici di comparazione** per visualizzare le prestazioni dei diversi modelli addestrati o pre-addestrati, facilitando l'analisi e l'identificazione degli algoritmi più efficaci per ciascun ambiente.

### 2. Setup dell'Ambiente
Prima di eseguire il codice, assicurati di utilizzare la versione di **Python 3.10.\***. È importante configurare correttamente l'ambiente seguendo questi passaggi:
1. **Creare un Ambiente Virtuale**:
    *   Apri il terminale o il prompt dei comandi.
    *   Esegui il comando: `python -m venv venv`
2. **Attivare l'Ambiente Virtuale**:
    *   Su Windows: `.\venv\Scripts\activate`
    *   Su Unix o MacOS: `source ./venv/bin/activate` oppure `source ./venv/Scripts/activate`
3. **(Opzionale) Disattivare l'Ambiente Virtuale**:
    *   Esegui il comando: `deactivate`
4. **Installare le Dipendenze**:
    *   Dopo aver clonato il progetto e attivato l'ambiente virtuale, installa le dipendenze richieste utilizzando: `pip install -r requirements.txt`. Questo comando scaricherà tutti i moduli non standard necessari.
5. **Aggiornare pip (se necessario)**:
    *   Se la versione di pip non è aggiornata, esegui: `pip install --upgrade pip`.

### 3. Struttura della Repository
Il repository è organizzato per facilitare la navigazione e la gestione dei file. Ed è presentato di seguito:

```
├── results/               # Cartella per il salvataggio dei risultati
│   ├── evaluation/        # Cartella per il salvataggio delle performance degli agenti negli ambienti specificati
│   │   ├── metrics_Ant-v5.json
│   │   ├── metrics_Hopper-v5.json
│   │   └── ...
│   ├── hyperparameters/   # Cartella per il salvataggio degli iperparametri migliori per i modelli specifici per gli ambienti
│   │   ├── A2C_Ant-v5_best_params.json
│   │   ├── DQN_Andt-v5_best_params.json
│   │   └── ...
│   ├── model/             # Cartella per il salvataggio dei modelli preaddestrati
│   │   ├── A2C_Ant-v5.zip
│   │   ├── DQN_Ant-v5.zip
│   │   ├── DQN_Hopper-v5.zip
│   │   └── ...
│   ├── plots/             # Cartella per il salvataggio dei grafici
│   │   ├── Comparison of Algorithms rewards during 10 episodes for Ant-v5.png
│   │   ├── Rewards during episodes for Ant-v5.png
│   │   └── ...
│   └── videos/            # Cartella per il salvataggio dei video utili alla visualizzazione del modello in esecuzione
│
├── monitoring/            # Cartella per il monitorning dell'ambiente durante l'addestramento
│   
├── Vec_normalization/     # Cartella per il salvataggio delle statistiche di normalizzazione di un ambiente
├── argumentParser.py
├── environments.py
├── main.py                # File da eseguire per l'esecuzione del programma
├── model.py
├── plotter.py
├── requirements.txt
└── README.md
```

### 4. Agenti Supportati  
Questo progetto supporta l'addestramento e la valutazione dei seguenti agenti (elencati in ordine di complessità) forniti dall'ambiente Gymnasium:  
*   **Hopper-v5**: agente monopode 2D progettato per testare algoritmi di equilibrio e salto. L'obiettivo è mantenere un'oscillazione ritmica in avanti senza sovraccaricare le articolazioni, con osservazioni focalizzate su angolazione del torso e velocità lineari/angolari.  
*   **HalfCheetah-v5**: Ambiente di simulazione basato su MuJoCo che modella un agente bidimensionale simile a un ghepardo. L'obiettivo è ottimizzare la locomozione per raggiungere la massima velocità anteriore, controllando torque continui su 6 articolazioni. La versione v5 introduce fisiche più stabili e osservazioni più dettagliate rispetto alle iterazioni precedenti.  
*   **Ant-v5**: agente quadrupede 3D che richiede il coordinamento di 8 giunti per una locomozione efficiente. La sfida principale è bilanciare stabilità dinamica e consumo energetico, con osservazioni che includono dati cinetici e di contatto con il terreno.  
*   **Humanoid-v5**: Ambiente avanzato per il controllo di un umanoide bipede con 21 gradi di libertà. L'agente deve apprendere strategie di camminata evitando cadute, gestendo complesse interazioni fisiche tra arti e torso. La versione v5 migliora la gestione dei collisioni e riduce gli artefatti di simulazione.  

Tutti gli ambienti utilizzano MuJoCo 3.0 e offrono spazi d'azione continui, rendendoli ideali per testare algoritmi di reinforcement learning su problemi di controllo motorio ad alta dimensionalità.

### 5. Algoritmi Implementati
Questo progetto include le implementazioni dei seguenti algoritmi di Reinforcement Learning:
*   **Proximal Policy Optimization (PPO)**: Un metodo di policy gradient che utilizza una funzione obiettivo surrogata per consentire aggiornamenti in mini-batch su più epoche. PPO offre un buon equilibrio tra prestazioni e facilità di implementazione.
*   **Soft Actor-Critic (SAC)**: Un algoritmo actor-critic off-policy che mira a massimizzare sia la ricompensa attesa che l'entropia della policy, incoraggiando l'esplorazione e la robustezza.
*   **Actor-Critic (A2C)**: Una variante di actor-critic che esegue aggiornamenti paralleli su più agenti (o copie dell'ambiente) per stabilizzare l'apprendimento.
*   **Deep Q-Network (DQN)**: Un algoritmo value-based che utilizza una rete neurale profonda per approssimare la funzione Q, imparando la migliore azione da intraprendere in ogni stato discreto. Lavorando in uno stato discreto, quando questo algoritmo viene specificato il programma procede ad una discretizzazione dello spazio delle azioni.
*   **Twin Delayed Deep Deterministic Policy Gradient (TD3)**: Un algoritmo actor-critic off-policy che affronta il problema della sovrastima della funzione Q utilizzando due critici e aggiornamenti ritardati della policy.
*   **Random Policy**: Una policy di base che seleziona azioni casualmente, utile per confrontare le prestazioni degli algoritmi di apprendimento.

### 6. Utilizzo del Progetto

#### Configurazione dei Parametri
L'utente potrà configurare vari aspetti dell'esecuzione del progetto tramite argomenti da riga di comando. I parametri configurabili sono visualizzabili eseguendo il seguente codice
```bash
python main.py --help
```
Per chiarezza sono riportati anche di seguito:
*   ```--model_type```: Tipo di modello. Sono accettati i seguenti parametri: ```PPO```, ```DQN```, ```A2C```, ```TD3```, ```SAC```, ```random```. Se non specificato il parametro di devault è ```PPO```
*   ```--env_id```: ID dell'ambiente gymnasium. Sono accettati i parametri ```Humanoid-v5```, ```HalfCheetah-v5```, ```Hopper-v5```, ```Ant-v5```. Default ```Ant-v5```
*   ```--save_path```: Percorso per il salvataggio dei modelli preaddestrati. Attenzione, se non si vogliono sovrascrvere i modelli preaddestrati questo parametro deve essere specificato. Default ```./results/model/```
*   ```--save_eval_path```: Percorso per il salvataggio delle metriche di valutazione. Attenzione, se non si vogliono sovrascrivere le metriche dei modelli preaddestrati questo parametro deve essere modificato. Default ```./results/evaluation```
*   ```--total_timesteps```: Numero dei time steps con il quale addestrare il modello. Default ```500_000```
*   ```--device```: Device per il running del modello. Per modelli non basati su architetture neuronali è preferibile specificare il parametro ```cpu``` per un'esecuzione più rapida. Per l'esecuzione su scheda grafica specificare ```cuda```. Default ```cuda```
*   ```--tunehp```: Se specificato viene eseguito l'hyperparameter tuning
*   ```--no_train```: Se specificato non viene eseguito l'addestramento
*   ```--n_envs```: Numero di ambienti paralleli inizializzati. Default 1
*   ```--seed```: Seed casuale per la riproducibilità dei risultati. Default 42
*   ```--env_monitor_dir```: Percorso per la directory in cui salvare i dati di monitoraggio dell'ambiente. Default ```./monitoring/```
*   ```--no_record_video```: Se specificato non viene eseguito il rendering dei video
*   ```--evaluate_model```: Se specificato viene eseguita la valutazione dei modelli. Per completezza di confronto è preferibile specificarlo solo quando sono stati addestrati tutti i modelli per l'ambiente specificato
*   ```--n_eval_episode```: Numero di episodi da eseguire per la valutazione dei modelli. Default 10
*   ```--comparison_plot```: Se specificato vengono graficati i confronti tra i modelli. Il confronto richiede la precedente valutazione dei modelli
#### Esecuzione dell'ottimizzazione degli iperparametri
L'utente può specificare se effettuare l'ottimizzazione degli iperparametri. Alcuni iperparametri sono già stati calcolati e sono contenuti nella directory ```./results/hyperparameters/```. Per l'esecuzione della sola fase di hyperparameter tuning è sufficiente che l'utente esegua il seguente codice:
```bash
pyton main.py --tunehp --no_train --no_record_video --model_type <preferred_model_type> --env_id <preferred_environments> 
```

#### Esecuzione dell'Addestramento
Una volta calcolati e salvati gli iperparamtetiri, l'utente potrà eseguire il processo di addestramento per l'agente e l'algoritmo specificati. Lo script si occuperà di creare l'ambiente, inizializzare l'agente con l'algoritmo selezionato e avviare il ciclo di addestramento. È preferibile che l'utente scelga un numero di time steps maggiore di quello specificato di default specialmente per agenti più complessi come l'Ant-v5 o l'Humanoid-v5.

Per l'esecuzione dell'addestramento l'utente può eseguire il seguente codice
```bash
python main.py --no_record_video --env_id <preferred_environment> --model_type <preferred_model_type> --total_timesteps <number_of_timesteps>
```

#### Valutazione degli Agenti e Presentazione dei Grafici
Dopo l'addestramento (o il caricamento di un modello pre-addestrato), l'utente potrà eseguire una fase di valutazione. Per la valutazione dei modelli è consigliato di specificare l'ambiente nel quale valutarli. Durante la valutazione, l'agente addestrato interagirà con l'ambiente per un certo numero di episodi che può essere specificato dall'utente, e verranno calcolate metriche di performance per quantificare l'efficacia dell'apprendimento. Verrà effettuato un confronto con una policy casuale per evidenziare il miglioramento ottenuto.

Per l'esecuzione della valutazione si può eseguire il seguente codice
```bash
python main.py --no_train --evaluate_model --env_id <preferred_evaluating_environment_id>
```
A seguito della fase di valutazione è possibile visualizzare i grafici di comparazione integrando il precedente codice con `--comparison_plot` oppure eseguendo il seguente codice
```bash
python main.py --no_train --no_record_video --comparison_plot --env_id <preferred_evaluating_environment_id>
```
I grafici che vengono presentati in questo progetto sono:
* **Grafici della ricompensa media**: un boxplot della ricompensa media ottenuta dagli agenti nei diversi episodi di valutazione
* **Andamento della ricompensa**: un grafico che mostra l'andamento della ricompensa durante gli episodi di valutazione

Si noti che la valutazione comprende sempre l'esecuzione di una policy casuale. Questo favorisce il confronto tra i modelli.


#### Utilizzo di Modelli Pre-Addestrati
Il progetto fornirà una selezione di modelli pre-addestrati per alcuni agenti e algoritmi. L'utente potrà specificare di non voler eseguire l'addestramente e in questo caso il programma caricherà il modello pre-addestrato specificato. In questo caso, se specificato procederà direttamente alla fase di valutazione e generazione dei grafici di comparazione.

ATTENZIONE: alcuni degli algoritmi utilizzati in questo progetto come SAC o A2C possono richiedere lunghi tempi di addestramento. In particolare, per l'addestramento i modelli preaddestrati ivi presentati sono stati addestrati tutti per un totale di 2000000 (2 mln) di time steps e hanno impiegato da 1h e 20min (PPO) a 9h e 45min (SAC).
