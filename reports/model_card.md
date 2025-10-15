# Model Card: Wine Type Classifier

## Model Details
- **Algorithm**: Random Forest Classifier
- **Version**: 1.0
- **Date**: September 2025
- **Author**: Nela Antropova

## Intended Use
- **Primary**: Classify wine as red or white based on chemical properties
- **Users**: Quality control teams, wine producers
- **Out-of-scope**: Wine quality grading, origin detection

## Performance
- **Test Accuracy**: 0.9969
- **ROC-AUC**: 0.9999
- **False Positives**: 2
- **False Negatives**: 2

## Training Data
- **Source**: UCI Wine Quality Dataset
- **Size**: 6,497 samples
- **Features**: 11 physicochemical properties
- **Class Balance**: 25% red, 75% white

## Limitations
- Geographic constraint (Portuguese wines)
- May not generalize to non-traditional wine styles
- Requires accurate chemical measurements

## Ethical Considerations
- No personal data used
- Transparent, interpretable model
- Human oversight recommended initially
