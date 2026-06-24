# DirectNet – detectarea anomaliilor hiperspectrale

Implementare locală DirectNet pentru detectarea nesupravegheată a anomaliilor din imagini hiperspectrale.

Ground truth-ul este opțional. Scriptul poate rula doar cu un cub HSI și generează hărți de scor și predicții pe percentile.
## Ideea metodei

Pentru fiecare pixel se extrage un patch. Zona centrală este ascunsă, iar rețeaua reconstruiește pixelul central din context.

- eroare mică → probabil fundal;
- eroare mare → anomalie posibilă.

Modelul nu recunoaște semantic obiecte, ci regiuni greu de prezis din vecinătate.
## Structura proiectului
```text
DirectNet_local_project_optional_gt/
├── DirectNet_03_experimente_optional_gt.py
├── pavia.mat
├── requirements.txt
└── README.md
```
Fișierul `.mat` trebuie pus lângă script.
## Cerințe

- Python 3.10 sau 3.11;
- minimum 8 GB RAM;
- GPU NVIDIA recomandat.
## Instalare
### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
În VS Code selectează interpreterul `.venv`.
## Rulare
```bash
python DirectNet_03_experimente_optional_gt.py
```
## Configurarea fișierului
```python
DATA_FILENAME = "pavia.mat"
CUBE_KEY = None
CUBE_LAYOUT = "auto"
```
- `DATA_FILENAME` = numele fișierului local;
- `CUBE_KEY = None` = alege automat cel mai mare array numeric 3D;
- `CUBE_LAYOUT = "auto"` = încearcă orientarea automată.

Dacă orientarea este greșită, folosește una dintre valorile:
```text
HWB, BHW, HBW, WHB, WBH, BWH
```
Exemplu:
```python
CUBE_LAYOUT = "BHW"
```
## Ground truth opțional
```python
USE_GROUND_TRUTH_IF_AVAILABLE = True
GROUND_TRUTH_KEY = None
```
Dacă este găsită o hartă 2D compatibilă, se calculează:

- ROC-AUC și PR-AUC;
- Precision, Recall și F1;
- FP și FN;
- pragul Youden.

Dacă nu există ground truth, scriptul continuă și folosește:
```python
DEFAULT_THRESHOLD_PERCENTILE = 99.5
```
La percentila `99.5`, aproximativ cei mai mari `0.5%` dintre pixeli sunt marcați ca anomalii candidate.

Fără ground truth nu se poate verifica automat dacă o detecție este corectă.
## Mod rapid
```python
QUICK_MODE = True
```
Pentru experimente finale:
```python
QUICK_MODE = False
```
## Parametri principali
```python
BATCH_SIZE = 100
INFERENCE_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WIN_VALUES = [1, 3, 5, 7, 9]
WOUT_VALUES = [15, 19, 23]
REFERENCE_WOUT = 19
```
- `Win` = zona centrală ascunsă;
- `Wout` = dimensiunea patch-ului exterior.
## Rezultate generate
```text
directnet_experiments/
├── checkpoints/
├── figures/
├── best_score_map.npy
├── best_prediction_map.npy
├── top_anomaly_candidates.csv
├── threshold_analysis.csv
├── win_sweep_results.csv
└── wout_sweep_results.csv
```
Se creează și `directnet_experiments.zip`.

Fără ground truth, configurația finală este aleasă prin `tail_contrast`, un indicator euristic al separării scorurilor foarte mari. Nu reprezintă acuratețea reală.
## Probleme frecvente
### Fișierul nu este găsit

Verifică `DATA_FILENAME` și poziționarea fișierului.
### Memorie insuficientă
```python
BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 64
```
### Orientare greșită

Setează manual `CUBE_LAYOUT` și, dacă este necesar, `CUBE_KEY`.
### Rulare pe CPU

Codul funcționează, dar antrenarea va dura mai mult.
