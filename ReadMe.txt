Coding_Directory_Template/
├── Code/                
│   ├── SourceCode/      # source code
│   └── classes/         # classes and other functions
│   └── run_scripts/     # batch scripts and files for running code on Slurm/PBS systems
├── Input/
│   ├── Data/            # data sources
│   └── Model/           # numerical model source code and output
└── Output/
    ├── Data/             # generated/processed data
    └── Figures/          # plots, animations

On Local Machine:
1. Create project-specific folder.
2. Copy "Coding_Directory_Template" folder into project folder.

For GitHub:
1. Store .git here in the "Coding_Directory_Template" folder.
2. Add textfile .gitignore here with:
    .DS_Store
    .ipynb_checkpoints
    /Input/Data/**
    /Input/Model/**
    /Output/Data/**
    /Output/Figures/**
    *.json
3. Repo name can be the name of the project-specific folder.