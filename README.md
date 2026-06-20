# HTB Progress Dashboard

Custom local dashboard for Hack The Box profile progress and friend comparison.

## What it shows

- Global ranking, points, rank, rank ownership, respects, user owns, system owns, bloods
- Friend comparison table
- Activity feed

## Setup

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

If script execution is blocked in PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set:

```env
HTB_API_TOKEN=your_token_here
```

Run the app:

```bash
streamlit run app.py
```

## Add more friends

Edit `.env`:

```env
HTB_USER_IDS=2266673,3769,4935,NEW_ID_HERE
```

Or type IDs in the sidebar at runtime.

