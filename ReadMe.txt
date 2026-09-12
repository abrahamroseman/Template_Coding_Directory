Coding_Directory_Template/
├── Code/                
│   ├── SourceCode/      # source code
│   └── classes/         # classes and other functions
│   └── run_scripts/     # batch scripts and files for running code on Slurm/PBS systems
├── Input/
│   ├── data/            # data sources
│   └── Model/           # numerical model source code and output
└── Output/
    ├── data/             # generated/processed data
    └── Figures/          # plots, animations

For GitHub:
1. Store .git here.
2. Add textfile .gitignore with:
    .DS_Store
    .ipynb_checkpoints
    /Input/data/**
    /Input/Model/**
    /Output/data/**
    /Output/Figures/**
    *.json