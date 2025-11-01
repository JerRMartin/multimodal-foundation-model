
# Multimodal Foundation Model
The goal of this project is to use foundation models from [huggingface](https://huggingface.co) to add and train additional small layers for computer vision recognition of a few Non-Verbal Cues (NVCs). 

Additionally, to take this refined output and convert it to audio descriptions using a pre-trained tts model. 

## Running the Project
### 1. Create a virtual environment named `.venv` (safe local name) and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Notes:
- If `python3` isn't available, try `python`. Check the Python version with `python --version`.

### 2. Install the project requirements.

```bash
pip install -r requirements.txt
```

### 3. *OPTIONAL: Run the automated tests (pytest).*

```bash
python -m pytest -q
```

Expected output is that tests pass (exit code 0) and you see a brief pytest summary.

### 4. Run the package entrypoint.

```bash
python -m src.main
# OR
python src/main.py
```

## Notes for Windows native (PowerShell / CMD)

- The above instructions assume you are running inside WSL. If you prefer to run natively on Windows (PowerShell), create and activate a venv with:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m pytest -q
```

## License
This repository is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0).

You may use, share, and adapt this work for **non-commercial research and educational** purposes, with proper attribution.
For commercial licensing inquiries, please contact [JeremyRobertMartin@gmail.com](mailto:jeremyrobertmartin@gmail.com).
