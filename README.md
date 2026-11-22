# European AI Championship Paris 2026, équipe kame_network
 
Code de l'équipe kame_network pour la manche de Paris de l'European AI Championship (PoC Innovation × CISPA, Epitech Le Kremlin-Bicêtre, 21 et 22 novembre 2026). Hackathon de 24 h en trois tâches, avec le calcul sur le supercalculateur JURECA (Jülich Supercomputing Centre) via Slurm.
 
Équipe : Kamil Mandi, Anis Gherib, Erwan Biscos, Mohamed-Souleyman Mechichi.
 
| Tâche | Responsables | Dossier |
|---|---|---|
| Model Equality : retrouver quels endpoints servent le même modèle | Erwan Biscos, Mohamed-Souleyman Mechichi | `src/model_equality/` |
| Unlearning : faire oublier 3 500 images à un ViT-B/16 | Kamil Mandi, puis Mohamed-Souleyman Mechichi | `src/unlearning/` |
| Watermark : localisation de watermarks | Anis Gherib | voir la section Watermark |
 
## Structure
 
```
src/model_equality/   collecte sur les endpoints, statistiques, regroupement, fichier de soumission
src/unlearning/       entraînement, cibles « gold », extrapolation, évaluation hors ligne, jobs Slurm
tests/                tests des propriétés revendiquées dans le code
docs/cours/           note de méthode sur Model Equality (LaTeX + PDF)
```
 
Les fichiers fournis par les organisateurs (`task_template.py`, `submission.py`, `query_reference.py`), le modèle, les images et les checkpoints ne sont pas versionnés. Aucune clé n'est dans le dépôt : les scripts la lisent dans une variable d'environnement (`ME_KEY` pour Model Equality) ou dans un fichier local ignoré par git.
 
## Installation
 
```bash
pip install -r requirements.txt
pytest
```
 
## Model Equality
 
Les endpoints notés acceptent 100 réponses chacun, les endpoints de référence sont illimités. La démarche :
 
1. Batteries de questions courtes à réponse unique, regroupées par paquets de 9 dans une même requête pour économiser le budget, et formulées de plusieurs façons pour résister au watermark.
2. Familles de modèles séparées par la limite de 1 024 tokens d'entrée (textes frontières calibrés sur plusieurs tokenizers).
3. Distance entre endpoints : ½ MMD² sur les réponses en one-hot (estimateur sans biais), test de permutation exact avec correction de Holm.
4. Regroupement par liaison complète à l'intérieur de chaque famille, choix du nombre de groupes par silhouette et stabilité par bootstrap.
Le détail des notations et des preuves est dans `docs/cours/01_model_equality.pdf`.
 
## Watermark
 
Tâche menée par Anis Gherib : localiser le watermark invisible dans des images (lecture recommandée par les organisateurs : Pixel Seal, Meta). Son code sera ajouté dans `src/watermark/`.
 
## Unlearning
 
Modèle : `vit_base_patch16_224` (timm), 100 classes. Il faut lui faire oublier 3 500 images sans labels. Le score vaut oubli × rétention, avec des distributions à T = 2 :
 
- oubli : la perte du modèle soumis sur les images oubliées (moyenne et écart-type) et ses prédictions doivent ressembler à celles de modèles « gold » réentraînés sans ces images ;
- rétention : rapport d'accuracy et KL(original ‖ soumis) sur des données conservées.
### Démarche
 
1. **Fine-tuning vers une cible approchée** (`unlearn.py`). Premier proxy de gold : les prédictions de l'original sur des vues fortement augmentées des images à oublier. On extrapole ensuite les poids, w = orig + α (sub − orig), avec `interp.py`.
2. **Données externes pour la rétention** (`prep_ext.py`). 30 000 images de mini-ImageNet (autorisées par les organisateurs), sur lesquelles on garde les sorties de l'original (KL à T = 2). Les 1 000 premières servent de holdout jamais utilisé à l'entraînement.
3. **Estimer gold avec des « surrogates »** (`surrogate.py`). On part d'une copie de l'original, on rapproche ses poids d'une initialisation aléatoire (shrink & perturb, plus fort sur les blocs hauts que sur les blocs bas), puis on la redistille sur les images externes uniquement. On garde un surrogate seulement si les images à oublier ne sont pas plus faciles pour lui que des images externes de mêmes classes (critère de fuite affiché par le script).
4. **Autres estimateurs** : kNN sur les features de l'original (`gold_knn.py`, `gold_knn2.py`), combinaison surrogate + kNN pondérée par image (`combine_gold.py`), fusion par classe (`ms_fuse_targets.py`).
5. **Entraînement vers la cible gold** : 4·KL(p_gold ‖ p_sub) à T = 2 sur les images à oublier et des copies légèrement altérées, plus un petit terme à T = 1 (`--lam_t1`) pour garder une perte dispersée, et la rétention sur les images externes.
6. **Variantes finales** : plusieurs graines, moyenne des poids de plusieurs modèles (soupe) et extrapolation α = 1.5 (`job_ms_seeds.sh`, `ms_auto.py`).
### Scores publics (30 % des données)
 
| Soumission | Cible | Score |
|---|---|---|
| `sub_v6_none_r0.3`, α = 1.3 | original sur vues augmentées | 0.0308 |
| `sub_v10_lr2e-4_ext1` | 2 surrogates (shrink & perturb sur les blocs hauts) | 0.0520 |
| `sub_v13_mix4` | 4 surrogates | 0.0596 |
| `sub_v13_v3` | 2 surrogates sans fuite (shrink & perturb en deux étages) + terme T = 1 | 0.0730 |
| meilleure soumission de l'équipe | variantes de `v13_v3` (graines, soupe, extrapolation) | 0.0987 |
 
### Outils
 
- `eval_sub.py`, `calib.py` : évaluation hors ligne d'une soumission (CE sur les images oubliées, accord et KL sur le holdout externe), sans consommer de soumission.
- `submit_queue.py` : file de soumissions (une toutes les 9 minutes), avec vérification du format avant envoi et journal CSV des scores.
- `job_*.sh` : jobs Slurm pour JURECA. Ils attendent la variable `UL`, le dossier de travail qui contient le modèle, les images et `ext_retain.npy` :
```bash
export UL=/chemin/vers/le/dossier/unlearning
sbatch src/unlearning/job_v13.sh
```
 
Règle du cluster pendant l'événement : un seul nœud de 4 GPU à la fois pour toute l'équipe, jobs en attente compris.
 