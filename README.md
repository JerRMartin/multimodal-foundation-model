
# Multimodal Foundation Model
The goal of this project is to use foundation models from [huggingface](https://huggingface.co) to add and train additional small layers for computer vision recognition of a few Non-Verbal Cues (NVCs). 

Additionally, to take this refined output and convert it to audio descriptions using a pre-trained tts model. 

## System Requirements
### CUDA Runtime
This project requires CUDA `12.8` or newer.

Make sure your installed NVIDIA driver supports at least CUDA `12.8`; otherwise, PyTorch may not detect your GPU.

#### Checking your Version
You can check this with the following conole command — look for the *“CUDA Version”* field. 

```bash
nvidia-smi
```

Example Output: *(Note the CUDA Version in the top-right)*
```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.65                 Driver Version: 577.03         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   48C    P8              3W /  120W |     162MiB /   8188MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
```

## Running the Project
### 1. Create a virtual environment named `.venv` (safe local name) and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Notes:
- If `python3` isn't available, try `python`. Check the Python version with `python --version`.

- For this Project it is recommended to use `3.11.*` as many dependencies require this version. 

### 2. Install the project requirements.

```bash
pip install -r requirements.txt
```

### 3. *OPTIONAL: Run the automated tests (pytest).*

```bash
python -m pytest -q
```

Expected output is that tests pass (exit code 0) and you see a brief pytest summary.

---

### Notes for Windows native (PowerShell / CMD)

- The above instructions assume you are running inside WSL. If you prefer to run natively on Windows (PowerShell), create and activate a venv with:

#### PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

#### CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m pytest -q
```

## Running the Project

```bash
python -m src.main
```
> **Optional Flags/Command-line Argument**
>
> `--control` - include this flag if you are using an Xbox Controller for Haptic Feedback.

## License
This repository is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0).

You may use, share, and adapt this work for **non-commercial research and educational** purposes, with proper attribution.
For commercial licensing inquiries, please contact [JeremyRobertMartin@gmail.com](mailto:jeremyrobertmartin@gmail.com).
