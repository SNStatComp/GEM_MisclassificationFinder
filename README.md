# GEM_MisclassificationFinder
This repository implements the generalized EM (GEM) algorithm, a method for identifying misclassified labels using auxiliary data and background variables. 

The code is accompanying the paper **Identifying misclassified labels in Register Data** (submitted). 

It includes an example using the dry beans dataset used in [Koklu, Ozkan (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0168169919311573). 

## Repository structure
```text
project-root/
│
├── config/                # Central configuration folder
│   ├── models.toml        # Configuration file for ML model selection
│   └── dry_beans.toml     # Central configuration file for EM algorithm
│
│
├── dry_beans/             # Folder for dry beans data
│   ├── features.pickle 
│   └── labels.csv    
├── log/                   # Folder for log files
├ 
├── misc/                  # Folder for miscellaneous files (notebook images) 
│ 
├── results/               # Generated outputs (not code)
│   ├── confusion/      
│   ├── pi_stats/
│   ├── psi/
│   ├── results_df/        
│   └── Summary/
│
├── utils/                  # Helper functions
│   ├── bm25Vectorizer.py   # Text processing helper file        
│   ├── create_input_files.py   # Preprocess functions to create input files
│   ├── mainV3.py           # Main EM algorithm file
│   ├── sample.py           # Create sample helper function
│   ├── tau_functions.py    # Tau Functions file
│   └── utils.py            # Performance, likelihood computation functions                      
│
├── scenariosV3.py          # Entry point point (runs the full pipeline)
├── requirements.txt        # Package requirements
├── example_dry_beans.ipynb # Example notebook for adding misclassifications
├── export_results.ipynb    # Small script exporting and writing results
├── run_algorithm.ipynb     # Example notebook calling and optimizing labelquality EM algorithm
│      
└── README.md
```

## Installation 

Requires Python version 3.12.3.

Clone the repository: 

```bash
git clone https://github.com/SNStatComp/GEM_MisclassificationFinder.git 
cd GEM_MisclassificationFinder
``` 

Create and activate a virtual environment:

```bash
python -m venv .venv 

# Windows 
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt 
``` 

## Quick start

Run the example:

```bash
python utils/create_input_files.py -c dry_beans/Dry_Bean.csv
python scenariosV3.py
```

This creates the input files, runs the GEM algorithm on Dry Beans and saves the metric summary to `results/`.
For a step-by-step explanation see [example_dry_beans.ipynb](example_dry_beans.ipynb) (for constructing a suitable dataset) and [run_algorithm.ipynb](run_algorithm.ipynb) (to run the algorithm on Dry Beans). 

## Data

The example uses the Dry Beans dataset, available for download on [Kaggle](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset). The dataset must be downloaded, where `Dry_Bean.csv` should be placed in the `dry_beans/` folder. The `cm_drybeans.csv` is constructed using the results of [Koklu, Ozkan (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0168169919311573), where the average predictions of the four machine learning models per seed category is taken. The two files are used as input in `utils/create_input_files.py`, where `features_pickle` alongside `labels.csv` are generated for 10 different seeds. These input files alongside the configuration file `config/dry_beans.toml` and `config/models.toml` are then used for the GEM algorithm. 

The confidential data used in the paper cannot be shared. The public example demonstrates the method but does not reproduce the results obtained with those data.

## Citation 

If you use this work, please cite::

```bibtex
@article{delden2026_GEM,
  title   = {Identifying misclassified labels in Register Data},
  year    = {2026},
  doi     = {DOI},
  note    = {submitted}
}
```

## License 

This code is available under the EUPL-1.2 license. See [LICENSE](LICENSE) for details.

## Contact

For big questions or bug reports, please open an issue in this repository. 
