# Extracted fuzzy rule base

Inputs: Glucose, BMI, Age, Pregnancies - 3 terms each (Low, Medium, High).

## Final (simplified) rules

**Rule 1** (covers 45% of patients): IF Glucose is Low (~102 mg/dL) AND BMI is Low (~29 kg/m²) AND Age is Low (~27 yrs) AND Pregnancies is Low (~1.47) THEN risk = 0.03 (very low risk -> Non-diabetic)

**Rule 2** (covers 33% of patients): IF Glucose is Medium (~133 mg/dL) AND Pregnancies is Medium (~4.40) THEN risk = 0.35 (low risk -> Non-diabetic)

**Rule 3** (covers 5% of patients): IF Glucose is Medium (~133 mg/dL) AND BMI is High (~43 kg/m²) AND Age is Medium (~35 yrs) THEN risk = 0.85 (very high risk -> Diabetic)

**Rule 4** (covers 17% of patients): IF Glucose is High (~155 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is Medium (~35 yrs) THEN risk = 0.85 (very high risk -> Diabetic)

| rule | IF | THEN risk | meaning | decision | coverage | avg_weight |
|---|---|---|---|---|---|---|
| 1 | Glucose is Low (~102 mg/dL) AND BMI is Low (~29 kg/m²) AND Age is Low (~27 yrs) AND Pregnancies is Low (~1.47) | 0.033 | very low risk | Non-diabetic | 0.446 | 0.429 |
| 2 | Glucose is Medium (~133 mg/dL) AND Pregnancies is Medium (~4.40) | 0.352 | low risk | Non-diabetic | 0.333 | 0.310 |
| 3 | Glucose is Medium (~133 mg/dL) AND BMI is High (~43 kg/m²) AND Age is Medium (~35 yrs) | 0.853 | very high risk | Diabetic | 0.050 | 0.087 |
| 4 | Glucose is High (~155 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is Medium (~35 yrs) | 0.852 | very high risk | Diabetic | 0.172 | 0.174 |

## Linguistic terms (original units)

| feature | term | center | sigma | unit |
|---|---|---|---|---|
| Glucose | Low | 102.25 | 29.01 | mg/dL |
| Glucose | Medium | 132.73 | 15.82 | mg/dL |
| Glucose | High | 155.11 | 16.19 | mg/dL |
| BMI | Low | 28.70 | 7.24 | kg/m² |
| BMI | Medium | 36.99 | 6.46 | kg/m² |
| BMI | High | 43.01 | 5.35 | kg/m² |
| Age | Low | 26.93 | 12.20 | yrs |
| Age | Medium | 35.00 | 10.71 | yrs |
| Age | High | 56.25 | 3.05 | yrs |
| Pregnancies | Low | 1.47 | 3.79 |  |
| Pregnancies | Medium | 4.40 | 3.65 |  |
| Pregnancies | High | 10.40 | 2.05 |  |

## Rules before simplification

**Rule 1** (covers 54% of patients): IF Glucose is Low (~102 mg/dL) AND BMI is Low (~29 kg/m²) AND Age is Low (~28 yrs) AND Pregnancies is Low (~1.47) THEN risk = 0.03 (very low risk -> Non-diabetic)

**Rule 2** (covers 6% of patients): IF Glucose is Medium (~133 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is Medium (~35 yrs) AND Pregnancies is Low (~1.47) THEN risk = 0.37 (low risk -> Non-diabetic)

**Rule 3** (covers 11% of patients): IF Glucose is Medium (~133 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is Medium (~35 yrs) AND Pregnancies is Medium (~4.40) THEN risk = 0.50 (uncertain -> Non-diabetic)

**Rule 4** (covers 5% of patients): IF Glucose is Medium (~133 mg/dL) AND BMI is High (~43 kg/m²) AND Age is Medium (~35 yrs) AND Pregnancies is Medium (~4.40) THEN risk = 0.81 (very high risk -> Diabetic)

**Rule 5** (covers 5% of patients): IF Glucose is Medium (~133 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is High (~56 yrs) AND Pregnancies is Medium (~4.40) THEN risk = 0.38 (low risk -> Non-diabetic)

**Rule 6** (covers 2% of patients): IF Glucose is Medium (~133 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is High (~56 yrs) AND Pregnancies is High (~10) THEN risk = 0.33 (low risk -> Non-diabetic)

**Rule 7** (covers 2% of patients): IF Glucose is High (~156 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is High (~56 yrs) AND Pregnancies is High (~10) THEN risk = 0.76 (high risk -> Diabetic)

**Rule 8** (covers 15% of patients): IF Glucose is High (~156 mg/dL) AND BMI is Medium (~37 kg/m²) AND Age is Medium (~35 yrs) AND Pregnancies is Medium (~4.40) THEN risk = 0.87 (very high risk -> Diabetic)

## Simplification log

- start: fidelity 0.958
- drop 'Glucose is Medium' from rule 6 -> fidelity 0.962
- drop 'Age is Medium' from rule 3 -> fidelity 0.963
- drop 'BMI is Medium' from rule 7 -> fidelity 0.963
- drop 'BMI is Medium' from rule 6 -> fidelity 0.962
- drop 'Pregnancies is Medium' from rule 5 -> fidelity 0.960
- drop 'Pregnancies is Medium' from rule 8 -> fidelity 0.958
- drop 'Pregnancies is Low' from rule 2 -> fidelity 0.960
- drop 'Pregnancies is Medium' from rule 4 -> fidelity 0.958
- drop 'BMI is Medium' from rule 3 -> fidelity 0.956
- drop 'Age is High' from rule 7 -> fidelity 0.954
- drop rule 2 -> fidelity 0.956
- drop rule 5 -> fidelity 0.959
- drop rule 5 -> fidelity 0.956
- drop rule 4 -> fidelity 0.957
