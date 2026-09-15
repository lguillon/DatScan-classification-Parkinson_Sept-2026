# Classification DatScan — TP

Classification binaire (sain / pathologique) de volumes DatScan (`.nii.gz`), à partir
d'un dataset hétérogène en résolution et protocole d'acquisition.

## Contenu

```
.
├── README.md
├── requirements.txt
├── notebook_classification.ipynb   # exploration, diagnostic et pipeline complet (référence)
├── src/
│   ├── preprocessing.py            # fonctions de prétraitement (crop physique + normalisation)
│   ├── preprocess.py               # script CLI : prétraite un zip brut en batch
│   ├── dataset.py                  # Dataset PyTorch
│   ├── models.py                   # SimpleCNN3D et SwinClassifier
│   ├── train.py                    # script CLI : entraînement + évaluation
│   └── predict.py                  # script CLI : prédiction sur de nouvelles données brutes
└── results/
    ├── best_cnn_model.pt           # poids du meilleur modèle CNN entraîné
    └── best_swin_model.pt          # poids du meilleur modèle Swin entraîné (si disponible)
```

## Installation

```bash
pip install -r requirements.txt
```

## 1. Prétraitement (à partir d'un zip brut de fichiers .nii.gz)

```bash
python src/preprocess.py \
    --zip_path /chemin/vers/data.zip \
    --output_dir ./data_prepped \
    --physical_size_mm 200 \
    --target_shape 128 128 128
```

Recadre chaque volume sur une taille physique fixe (en mm, indépendante du FOV
d'origine — le dataset source mélange plusieurs protocoles avec des champs de vue
très différents), le redimensionne vers une shape fixe en voxels, puis normalise
l'intensité par percentile (1-99), propre à chaque image. Reprenable : relancer la
même commande saute les fichiers déjà traités (voir `prep_log.csv` dans le dossier
de sortie).

## 2. Entraînement

```bash
python src/train.py \
    --data_dir ./data_prepped \
    --labels_csv /chemin/vers/labels.csv \
    --output_dir ./results \
    --model cnn \
    --epochs 20
```

`labels_csv` doit contenir au minimum deux colonnes : `uid` (nom du fichier sans
`.nii.gz`) et `is_pathologic` (0 ou 1).

Options principales :
- `--model {cnn,swin}` — `cnn` (rapide, baseline) ou `swin` (Swin Transformer 3D via
  MONAI, plus lourd)
- `--img_size D H W` — taille d'entrée du modèle (défaut 96 96 96)
- `--epochs`, `--batch_size`, `--lr` — hyperparamètres d'entraînement

Le script sauvegarde le meilleur modèle (sur accuracy de validation) dans
`<output_dir>/best_<model>_model.pt`, et affiche en fin d'exécution une matrice de
confusion, un rapport de classification (precision/recall/F1 par classe) et l'AUC —
l'accuracy seule n'est pas fiable vu le déséquilibre de classes du dataset (~55/45).

## 3. Prédiction sur de nouvelles données

```bash
python src/predict.py \
    --input_dir /chemin/vers/nouveaux_fichiers_bruts \
    --checkpoint ./results/best_cnn_model.pt \
    --model cnn \
    --output_csv ./results/predictions.csv
```

Prend des fichiers `.nii.gz` **bruts** (pas encore prétraités) en entrée — le
prétraitement est appliqué automatiquement avant la prédiction. Produit un CSV avec
`uid`, `predicted_label` et `prob_pathologic` pour chaque fichier.

## Notes méthodologiques

- **Hétérogénéité du dataset source** : le dataset mélange plusieurs
  protocoles/résolutions (champ de vue d'origine allant de ~175mm à ~630mm, spacing
  et nombre de coupes très variables). Un recalage affine vers un template T1
  (MNI152) a été testé mais abandonné : le signal DatScan (concentré sur le
  striatum, peu de structure anatomique ailleurs) ne fournit pas assez de repères
  pour un recalage par intensité fiable — cela produisait des rotations aberrantes.
  L'approche retenue (crop physique fixe + normalisation par percentile) est plus
  simple et robuste, au prix de ne pas corriger les différences d'orientation entre
  scans — partiellement compensé par de l'augmentation (flip) à l'entraînement.
- **Déséquilibre de classes** : ~55% pathologique / 45% sain — une Focal Loss est
  utilisée plutôt qu'une simple cross-entropy pour limiter le biais vers la classe
  majoritaire.
