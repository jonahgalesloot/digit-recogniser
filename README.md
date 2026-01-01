**Project Overview**

- **Purpose:** A minimal MNIST proof-of-concept repository that extracts MNIST data, trains small neural networks (CPU-only and optional NumPy/TQDM variant), and provides a simple browser-based drawing interface to load and run trained weights.
- To demo, a sample model is included in models/demo
**Included Files**

- `dataset_extracter.py`: Extracts MNIST IDX files into human-readable text bitmaps under `datasets/train/extracted`.
- `no_imports_train.py`: Original, CPU-only training script that uses only the Python standard library (no pip installs required). Slower, proof of concept
- `train.py`: Later written (depends on `numpy` and `tqdm`), faster and cleaner output, with progress bars, reccomended to run
- `interface.html`: Browser UI that lets you draw a 28x28 digit and load model layers (`layer_0.csv`, `layer_1.csv`, ...).
- `mouse_macro.py`: Utility script (included as-is) to stop PC from turning off
- `datasets/train/MNIST`: Raw IDX files
- `models/demo`: Pretrained demo model

**Prerequisites**

- Python 3.8+ installed.
- No extra packages required to run `dataset_extracter.py` or `no_imports_train.py`.
- If you want to run `train.py` with performance improvements, install `numpy` and `tqdm` (optional):

```bash
pip install numpy tqdm
```

**Quick Setup**

- Clone or download this repository.
- Ensure the raw MNIST files exist at `datasets/train/MNIST/images.idx3-ubyte` and `datasets/train/MNIST/labels.idx1-ubyte`.

**Step 1 — Extract MNIST bitmaps**

- Run the dataset extractor to convert IDX files into text bitmaps used by the trainers:

```bash
python dataset_extracter.py
```

- This will create `datasets/train/extracted/<digit>/image_<index>.txt` files (28 lines of 28 whitespace-separated 0/1 values), which are used by the trainers.

**Step 2 — Train a model (CPU-only, no external libs)**

- Use the included standard-library trainer for a dependency-free run:

```bash
python no_imports_train.py
```

- Notes:
  - The script prints generation progress in the console. It's a simple proof-of-concept and can be slow.
  - Training hyperparameters are defined near the top of `no_imports_train.py` (`population_size`, `generations`, `steps`, etc.). Edit them if you want fewer generations for faster runs.
  - At the end of training you'll be prompted for a model name. The script saves a folder under `models/<your_name>/` containing `input_to_hidden/neuron_*.csv` and `hidden_to_output/neuron_*.csv`.


**Step 3 — Using `interface.html`**

- Open `interface.html` in a modern browser (Chrome/Edge/Firefox).
- In the "Load Model Weights" section click the folder selector and choose the folder that contains `layer_0.csv` and `layer_1.csv` (or any `layer_N.csv` sequence matching layers in your model).
- Draw on the 28x28 grid (pen/eraser), predictions update automatically.



-- End of README --

