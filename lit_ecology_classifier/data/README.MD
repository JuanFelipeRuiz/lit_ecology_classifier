# data folder

Contains the data inside of threee diffrent folders:

```
└── data
    ├── interim        <- Intermediate data that has been transformed for eda and further tings.
    ├── processed      <- data for modeling, splitted by train/test/validation/OOD
    └── raw            <- original data, corresponds the enzipped download from eawags open source database eric
```

To prevent data leakege from the diffrent data set versions,  only one version split of the ZooLake dataset can be inside the processed folder for the modelling.