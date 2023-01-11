# Benchmark Standards
## Benchmark OBS Storage Structure
```
luojianet-benchmark
├── Scene_Classification                    // Five main task types
|   ├── VGG-16                              // Model name
|   |   ├── WHU-RS19                        // Dataset name
|   |   |   ├── Ascend                      // Device Type
|   |   |   |   ├── 1chip                   // Chip number
|   |   |   |   |   ├── code                // Model code
|   |   |   |   |   |   ├── train.py        // Training enter file     (Necessary)
|   |   |   |   |   |   ├── eval.py         // Evaluation enter file   (Necessary)
|   |   |   |   |   |   ├── pred.py         // Predicting enter file   (Necessary)
|   |   |   |   |   |   ├── ds_split.txt    // Dataset split txts      (Flexible)
|   |   |   |   |   |   ├── split.py        // Dataset split script    (Flexible)
|   |   |   |   |   |   ├── read_split.py   // Read split dataset file (Flexible)
|   |   |   |   |   |   ├── README.md       // Readme                 （Not necessary, just Recommend）
|   |   |   |   |   |   └── *               // Other model files
|   |   |   |   |   ├── ckpt                // CKPT save path
|   |   |   |   |   |   ├──seed_1           // Repeat
|   |   |   |   |   |   ├──seed_2
|   |   |   |   |   |   └── * 
|   |   |   |   |   ├── log                 // Training log save path
|   |   |   |   |   |   ├──seed_1           // Repeat
|   |   |   |   |   |   ├──seed_2
|   |   |   |   |   |   └── * 
|   |   |   |   |   └── other               // Other results save path (e.g. record txt)
|   |   |   |   |       ├──seed_1           // Repeat
|   |   |   |   |       ├──seed_2
|   |   |   |   |       └── * 
|   |   |   |   └── 8chips  
|   |   |   └── GPU                         // Device Type
|   |   |       ├── 1chip                   // Chip number
|   |   |       └── 8chips
|   |   └── *
|   ├── Resnet 
|   |   └── *
|   └── * 
├── Object_Detection
├── Semantic_Segmentaion
├── Change_Detection
├── 3D_Reconstucion
└── temp                        // Temp files
```

## Modelarts Experiments Naming Conventions
![](/experiment.png)
1. Training job has **NO** naming convention! Feel free!
2. Experiment must name as format **"{ModelName}_{DatasetName}_{1|8}chip{s}"** //keep only 2 '\_'