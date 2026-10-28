# Running the pipeline (all platforms)

Needs Python 3.9+ (tested 3.11; numpy 2.5, scipy 1.18, matplotlib 3.11). Runtime ~10 min on 2 cores.
Every platform ends with the same commands:  `python run_all.py`  then  `python verify.py`  (prints REPRODUCED).

## Windows (PowerShell)
    cd lightgenome
    py -m venv .venv ; .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python run_all.py ; python verify.py
(If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`.)

## macOS / Linux
    cd lightgenome
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_all.py && python verify.py        # or: make all

## Conda (any OS)
    conda env create -f environment.yml && conda activate lightgenome
    python run_all.py && python verify.py

## Docker
    docker build -t lightgenome .
    docker run --rm -v "$PWD/out:/app/results" lightgenome       # results appear in ./out
(For figures too, also mount `-v "$PWD/figs:/app/figs"`.)

## Google Colab / Kaggle / Jupyter
    !unzip -q lightgenome.zip
    %cd lightgenome
    !pip -q install -r requirements.txt
    !python run_all.py && python verify.py
    from IPython.display import Image; Image('figs/fig2_L1.png')

## HPC / SLURM
    #SBATCH -c 2 -t 00:30:00
    module load python && python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt && python run_all.py && python verify.py

## Reproducibility notes
* Seeds are fixed; results here matched bit-for-bit across two runs. A different numpy/scipy/BLAS build may change
  the last digits, so `verify.py` uses a 1e-6 tolerance (`python verify.py 1e-3` to loosen).
* `reference/` holds the results behind the reported numbers.
* Pieces: `python run_all.py --only validate|experiments|figures|summary|verify`; one experiment: `python experiments.py e1 e5`.
