# ParkedIn: demand-based dynamic pricing for urban parking

A multi-agent system that replays real parking data as if it were live. Every tick (one
30-minute slot), five agents run in turn and share results through a SQLite database:

| Agent | File | Method | Job |
| --- | --- | --- | --- |
| Data | `agents/data_agent.py` | pandas | Cleans the CSV, aligns readings to 30-minute slots, builds features, replays one slot at a time |
| Forecast | `agents/forecast_agent.py` | XGBoost (one model per step ahead) | Predicts each lot's occupancy for the next 6 slots (3 hours) |
| Pattern | `agents/pattern_agent.py` | K-Means, 4 clusters | Labels each lot's demand: low, medium, high, peak |
| Pricing | `agents/pricing_agent.py` | Rule-based start + grid search, one agent per lot | Picks each lot's next price inside its min/max limits |
| Monitor | `agents/monitor_agent.py` | Rolling forecast error | Raises alerts and retrains the forecast models by itself on drift |

`orchestrator.py` runs the agents every tick, `db.py` is the shared database, and
`dashboard/` is the Streamlit app (live map, lot details, admin).

## 1. One-time setup

You need **Python 3.10 to 3.13** and an internet connection (for installing packages and
for the map background).

```bash
# from inside the parkedin folder
python -m venv .venv

# activate it
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (cmd):          .venv\Scripts\activate.bat
# macOS / Linux:          source .venv/bin/activate

pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2. Get the dataset

Download `dataset.csv` (14 lots, 18,368 rows) and save it as `data/raw/dataset.csv`:

- Open https://github.com/sh-kartik18/Dynamic-Pricing-for-Urban-Parking-Lots/blob/main/dataset.csv
- Click the download button (Download raw file), then move the file into `data/raw/`.

Or from a terminal (macOS / Linux / Git Bash):

```bash
curl -L -o data/raw/dataset.csv https://raw.githubusercontent.com/sh-kartik18/Dynamic-Pricing-for-Urban-Parking-Lots/main/dataset.csv
```

## 3. Train the models (about 1 to 3 minutes)

```bash
python train.py
python evaluate_pricing.py
```

`train.py` builds the features, compares naive, ARIMA, Random Forest and XGBoost forecasts,
trains K-Means, and saves everything in `models/`. `evaluate_pricing.py` compares fixed
price, rules only, and rules + grid search over the whole test period. Both print tables you
can copy into the report; the Admin page shows them too.

## 4. Run the system (two terminals)

Activate the virtual environment in both terminals, in the `parkedin` folder.

```bash
# Terminal 1: the agents
python orchestrator.py

# Terminal 2: the dashboard (opens http://localhost:8501 in your browser)
streamlit run dashboard/Home.py
```

Press **Start simulation** on the Home page. Prices update every 5 seconds. Use the
**Admin** page to change the speed, the pricing balance, each lot's price limits, or to
reset the replay. Stop everything with `Ctrl + C` in both terminals.

## 5. Things to know

- **Start the orchestrator first.** It creates `parkedin.db`. To start completely fresh,
  stop both programs and delete `parkedin.db` (plus `parkedin.db-wal` / `-shm` if present).
- **After changing `config.yaml`**, restart the orchestrator. Price weights and speed can
  also be changed live on the Admin page.
- **After changing anything in the agents' training code**, run `python train.py` again.
- **Map not showing?** The map background needs internet. Everything else works offline.
- **Assumption:** the data never shows how drivers react to price, so the pricing agents and
  the revenue comparison use an assumed price sensitivity (`elasticity: 0.4` in
  `config.yaml`). `evaluate_pricing.py` also tests 0.2 and 0.6.
- **Data fixes made by the data agent:** readings are aligned to 30-minute slots, a few
  missing readings are interpolated, and one lot (`BHMMBMMBX01`) whose coordinates were far
  from the others is placed next to them on the map.

## 6. Put the dashboard online (Streamlit Community Cloud)

1. Push the repository to GitHub. Keep `requirements.txt` in the **repository root** (next to
   the `parkedin` folder); Streamlit Cloud only looks there or next to the main file.
2. On share.streamlit.io, create an app with main file path `parkedin/dashboard/Home.py`.
3. On the first visit the dashboard sets itself up: it downloads the dataset, trains the
   models (about 2 to 3 minutes) and starts the orchestrator in the background. Then press
   **Start simulation**. No second terminal is needed online.

## Results on the test days (11 days, 196 slots)

Forecast error (MAE of occupancy rate, lower is better):

| Slots ahead | Naive | ARIMA | Random Forest | XGBoost |
| --- | --- | --- | --- | --- |
| 1 | 0.0425 | 0.0450 | 0.0166 | **0.0156** |
| 3 | 0.1111 | 0.0961 | 0.0358 | **0.0350** |
| 6 | 0.1779 | 0.1330 | 0.0550 | **0.0538** |

Pricing at the assumed sensitivity of 0.4:

| Strategy | Revenue vs fixed Rs 40 | Avg distance from 85% full | Lot-slots 95%+ full | Avg price |
| --- | --- | --- | --- | --- |
| Fixed price | 0% | 0.346 | 137 | Rs 40.0 |
| Rules only | -9.3% | 0.293 | 0 | Rs 32.6 |
| Rules + grid search | **+2.3%** | 0.316 | **0** | Rs 42.1 |

## Project structure

```
parkedin/
├── agents/            five agents
├── dashboard/         Streamlit app (Home.py + pages/)
├── data/raw/          put dataset.csv here
├── data/processed/    created by train.py
├── models/            created by train.py
├── common.py          paths and config loading
├── config.yaml        all settings
├── db.py              SQLite helpers (the shared "blackboard")
├── orchestrator.py    runs one tick through all agents
├── train.py           trains and compares models
├── evaluate_pricing.py  compares pricing strategies
└── requirements.txt
```
