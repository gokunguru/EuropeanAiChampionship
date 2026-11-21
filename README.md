# European AI Championship Paris — 21–22 nov. 2026

Hackathon de 24 h.

- **Équipe** : _à compléter (noms + pseudos GitHub)_
- **Challenge** : _à compléter dès l'annonce du sujet_
- **Piste** : moteur d'induction de règles floues (relations floues d'ordre n, abduction)
- **Livrable** : une démo qui tourne + un pitch

## Structure

```
src/fuzzy_rules/   code du moteur (noms identiques aux Notations)
tests/             un test par propriété mathématique revendiquée
notebooks/         exploration et démo
```

## Installation

```bash
pip install -r requirements.txt
pytest
```

## Notations

Ce bloc est fixe : le code et les docs le reprennent tel quel. Tout changement passe par ici.

| Symbole | Type | Définition |
|---|---|---|
| `L = [0,1]` | ensemble | ensemble des degrés de vérité |
| `X_1, …, X_n` | ensembles finis | domaines des attributs |
| `X = X_1 × … × X_n` | ensemble | univers produit |
| `A : X_i → L` | fonction | sous-ensemble flou de `X_i`, de fonction d'appartenance `μ_A` |
| `R : X_1 × … × X_n → L` | fonction | relation floue d'ordre `n` |
| `T : L² → L` | t-norme | commutative, associative, croissante, neutre `1` |
| `S : L² → L` | t-conorme | duale de `T` : `S(a,b) = 1 − T(1−a, 1−b)` |
| `N : L → L` | négation | `N(a) = 1 − a` |
| `I_T : L² → L` | implication | résiduum de `T` : `I_T(a,b) = sup{ c ∈ L : T(a,c) ≤ b }` |
| `r = (A_1 ∧ … ∧ A_k ⇒ B)` | règle | prémisses `A_j`, conclusion `B` |

**À fixer (ambiguïté signalée, rien n'est encore tranché) :**

- Choix de `T` : `min` (Gödel), produit (Goguen) ou Łukasiewicz. Cela détermine `S` et `I_T`.
- Sens précis de « relation d'ordre n » : relation `n`-aire floue (hypothèse ci-dessus) ou relation d'ordre (préordre) flou. À confirmer.
- Définition du degré de vérité d'une règle sur un jeu de données (agrégation par `inf`, moyenne, ou autre).
