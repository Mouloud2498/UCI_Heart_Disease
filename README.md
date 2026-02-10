# ❤️ UCI Heart Disease — Pipeline de Prétraitement & Classification (Scikit-learn)

## 🎯 Objectif
Transformer des données brutes du dataset **Heart Disease** en un format exploitable et **reproductible** pour un modèle de classification, en automatisant :
- la gestion des **valeurs manquantes**
- l’**encodage** des variables catégorielles
- la **mise à l’échelle** des variables numériques
- l’entraînement/évaluation via un **Pipeline Scikit-learn** (avec `ColumnTransformer`)

📌 Ce projet met l’accent sur la **qualité des données**, l’**éviction des fuites d’information** (data leakage) et la **documentation** (comme attendu sur le marché canadien).

---

## 🧩 Contexte (Business / Problème)
Dans un contexte santé, la détection de signaux liés aux maladies cardiaques peut aider à orienter l’analyse clinique.  
Le défi est que les données contiennent des **valeurs manquantes**, des **variables catégorielles** et des **échelles différentes**, ce qui rend l’entraînement d’un modèle directement sur les données brutes peu fiable.

✅ **Problème à résoudre :** construire une chaîne de préparation robuste qui permet de comparer des variantes de prétraitement (StandardScaler vs RobustScaler) et de produire un modèle de référence.

---

## 📦 Dataset (Source)
Les données utilisées proviennent du dataset Kaggle **“Heart Disease Data”** :

- Source : https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

> 🔒 Le dataset n’est pas inclus directement dans ce dépôt.  
> Téléchargez-le depuis Kaggle puis placez-le dans `data/` (ou adaptez le chemin dans le script/notebook).

---

## 🧠 Variables & Cible
### Variables numériques (exemples)
- `age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`

### Variables catégorielles (exemples)
- `sex`, `dataset`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`

### Variable cible
- `num` : niveau de maladie (classification)

---

## 🛠️ Méthodologie de prétraitement (Justifications)

### 1) Imputation
- **Numériques** : imputation par la **médiane**  
  ✅ robuste face aux valeurs extrêmes (outliers) contrairement à la moyenne.
- **Catégorielles** : imputation par la **valeur la plus fréquente** (mode)  
  ✅ cohérent pour des modalités discrètes.

### 2) Encodage des catégorielles
- **`slope`** : encodage **ordinal** avec un ordre :  
  `downsloping < flat < upsloping`  
  ✅ car la variable a une hiérarchie naturelle.
- Autres catégorielles : **One-Hot Encoding** avec :
  - `drop='first'` : éviter la colinéarité (dummy trap)
  - `handle_unknown='ignore'` : robustesse si une modalité apparaît en test mais pas en train

### 3) Mise à l’échelle (comparaison)
Deux variantes évaluées :
- **StandardScaler** : centrage-réduction (moyenne 0, écart-type 1)
- **RobustScaler** : basé sur médiane + IQR  
  ✅ plus stable en présence d’outliers (ex : cholestérol)

### 4) Transformations complémentaires (analyse)
- **Discrétisation** (`KBinsDiscretizer`) : transformer une variable continue en intervalles pour analyse de profils.
- **PowerTransformer (Yeo-Johnson)** : réduire l’asymétrie (skewness) de `oldpeak`.
- **PolynomialFeatures (degré 2)** sur un sous-ensemble (ex : `age`, `trestbps`, `chol`) pour capturer des relations non linéaires sans explosion combinatoire.

---

## 🧱 Pipeline complet (ColumnTransformer + Pipeline)
Le prétraitement est encapsulé dans un `ColumnTransformer` :
- **Bloc numérique** : `SimpleImputer(median)` + (`StandardScaler` ou `RobustScaler`)
- **Bloc slope** : `SimpleImputer(most_frequent)` + `OrdinalEncoder`
- **Bloc nominal** : `SimpleImputer(most_frequent)` + `OneHotEncoder`

Le tout est enchaîné à un modèle **LogisticRegression** dans un **Pipeline unique** :
- ✅ reproductible
- ✅ évite la fuite d’information (transformations apprises uniquement sur train)

Le découpage `train_test_split` utilise `stratify=y` :
- ✅ conserve la proportion des classes (utile en cas de déséquilibre)

---

## 📏 KPI / Métriques d’évaluation
- **Accuracy** (métrique de base, utilisée ici)
- Recommandé pour aller plus loin (si déséquilibre) :
  - **Recall**, **F1-score**
  - matrice de confusion
  - ROC-AUC (optionnel)

---

## ✅ Résultats
Accuracy obtenue :
- **Pipeline StandardScaler** : **0.598**
- **Pipeline RobustScaler** : **0.603**

**Interprétation :** RobustScaler offre un gain léger, cohérent si certaines variables numériques contiennent des valeurs atypiques (outliers). Les performances restent modestes, ce qui suggère :
- analyse du déséquilibre de classes,
- métriques complémentaires (F1/Recall),
- modèles plus flexibles ou tuning.

---

## 🖼️ Visuels / Captures 

![Schéma pipeline](pretraitement_heart_disease.png)
