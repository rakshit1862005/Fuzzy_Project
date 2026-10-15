# Dataset: Pima Indians Diabetes

`diabetes.csv` is the Pima Indians Diabetes dataset, originally collected by the
US National Institute of Diabetes and Digestive and Kidney Diseases and
distributed through the UCI Machine Learning Repository and Kaggle.

- 768 patients (all female, of Pima Indian heritage, aged 21 or older)
- 8 numeric input features, 1 binary target (`Outcome`: 1 = diabetic)
- Class balance: 500 non-diabetic, 268 diabetic

| Column | Meaning | Unit |
|---|---|---|
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose 2 h after an oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skin-fold thickness | mm |
| Insulin | 2-hour serum insulin | µU/mL |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Score of diabetes history among relatives | – |
| Age | Age | years |

**Data quality note:** a value of `0` in Glucose, BloodPressure, SkinThickness,
Insulin or BMI is physically impossible and means "not measured". These zeros
are replaced by the training-set median (see `glassbox/data.py`).

Reference: Smith, J.W. et al. (1988). *Using the ADAP learning algorithm to
forecast the onset of diabetes mellitus.* Proc. Symposium on Computer
Applications in Medical Care, 261–265.
