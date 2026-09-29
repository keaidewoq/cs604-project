# cs604-project

Each person clones it, makes a branch for their work, and merges through a pull request.

Exact checkpoint: vit_base_patch16_224.augreg2_in21k_ft_in1k

cs604-project/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/                 # full fine-tuning, LoRA, ablations
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── notebooks/
│   └── run_on_colab.ipynb   # short setup and run commands
└── results/                 # small CSV summaries and plots