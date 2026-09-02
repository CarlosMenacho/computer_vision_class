# Computer Vision Class

Course materials, exercises, and tutorials for Computer Vision, based on
Szeliski's *Computer Vision: Algorithms and Applications* (2nd ed.).

## Environment Setup

This project uses `conda` for the base Python environment and `uv` for fast
package installation.

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate cv_class
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Verify the install

```bash
python -c "import cv2, numpy, rerun; print('OK')"
```

## Project Structure

```
excercices/   # Practice exercises
tutorials/    # Tutorial notebooks/scripts
slides/       # Course slide decks (.tex) and reference book
```

## Slides

https://drive.google.com/drive/folders/16FizqbGR7YJyOUBjx0GAiNnH3DyneTER?usp=drive_link

## Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python 3.12 (installed automatically via `environment.yaml`)
