# Jupyter Notebook Setup Guide

## ✅ Virtual Environment Kernel Setup

Your virtual environment has been registered as a Jupyter kernel named **"Python (CLV Prediction)"**.

## How to Use the Notebook with Your Venv

### Option 1: VS Code / Cursor

1. **Open the notebook** (`notebooks/01_eda.ipynb`)
2. **Select the kernel**: 
   - Click on the kernel selector in the top-right corner of the notebook
   - Choose **"Python (CLV Prediction)"** from the list
   - If you don't see it, click "Select Another Kernel..." → "Python (CLV Prediction)"

3. **Verify it's working**: The first cell should show the working directory when you run it.

### Option 2: Jupyter Notebook (Browser)

1. **Activate your virtual environment**:
   ```powershell
   cd customer_lifetime_value_prediction
   .\venv\Scripts\activate
   ```

2. **Start Jupyter Notebook**:
   ```powershell
   jupyter notebook
   ```

3. **Open the notebook** from the file browser

4. **Select kernel**: 
   - Kernel → Change Kernel → Python (CLV Prediction)

### Option 3: JupyterLab (Browser)

1. **Activate your virtual environment**:
   ```powershell
   cd customer_lifetime_value_prediction
   .\venv\Scripts\activate
   ```

2. **Start JupyterLab**:
   ```powershell
   jupyter lab
   ```

3. **Open the notebook** and select the kernel from the top-right

## Troubleshooting

### Issue: Kernel not showing up in IDE

**Solution**: The kernel is registered. Refresh your IDE or:
1. Close and reopen the notebook
2. In VS Code/Cursor: `Ctrl+Shift+P` → "Notebook: Select Notebook Kernel"

### Issue: "Jupyter command not found" in bash

**Solution**: You need to activate the venv first, then use the Python from venv:
```powershell
# Activate venv (PowerShell)
.\venv\Scripts\activate

# Then run jupyter
python -m jupyter notebook
```

Or directly:
```powershell
.\venv\Scripts\python.exe -m jupyter notebook
```

### Issue: ModuleNotFoundError even with correct kernel

**Solution**: Make sure you're using the correct kernel and restart it:
- Kernel → Restart Kernel → Restart

### Issue: Notebook hangs on first cell

**Solution**: This was fixed! The path manipulation issue has been resolved. If it still hangs:
1. Restart the kernel completely
2. Check that the working directory is set correctly (should print in first cell)

## Verify Installation

Run this in a Python terminal or notebook cell:
```python
import sys
print(sys.executable)
# Should point to: ...\venv\Scripts\python.exe
```

## Quick Commands Reference

```powershell
# Activate venv
.\venv\Scripts\activate

# Run Jupyter Notebook
jupyter notebook

# Run JupyterLab
jupyter lab

# List available kernels
jupyter kernelspec list

# Remove kernel (if needed)
jupyter kernelspec remove clv_prediction
```

