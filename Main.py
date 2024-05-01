import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel

# Load the dataset
data = pd.read_csv('C:/Users/Water/PycharmProjects/HitPredictorProject/archive/dataset-of-10s.csv')
X = data.drop(['track', 'artist', 'uri', 'target'], axis=1)
y = data['target']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X.columns)
])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selector', SelectFromModel(GradientBoostingClassifier(), threshold='median')),
    ('classifier', GradientBoostingClassifier())
])

# Hyperparameter tuning
param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 1.0]
}
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=3, scoring='precision', verbose=3, random_state=42, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Evaluating the best model
predictions = best_model.predict(X_test)
predicted_probs = best_model.predict_proba(X_test)[:, 1]
print("Best parameters:", random_search.best_params_)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, predicted_probs))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, predicted_probs)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, predicted_probs))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Define additional models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, kernel='linear', random_state=42)
log_reg = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)
gb = best_model  # using best_model for Gradient Boosting

rf.fit(X_train, y_train) # Fit RandomForest with training data
svc.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
# Voting Classifier setup
voting_clf = VotingClassifier(
    estimators=[('gb', gb), ('rf', rf), ('svc', svc), ('log_reg', log_reg)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)  # Fit the Voting Classifier on the training data

# Evaluating all models
models = [
    ('Gradient Boosting', best_model),
    ('Random Forest', rf),
    ('SVM', svc),
    ('Logistic Regression', log_reg),
    ('Voting Classifier', voting_clf)
]

model_performances = {}
for model_name, model in models:
    if not hasattr(model, 'predict_proba'):
        model.fit(X_train, y_train) # to ensure each model is fitted
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(pred)
    model_performances[model_name] = {
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': roc_auc_score(y_test, pred),
        'ROC AUC': roc_auc_score(y_test, prob)
    }

# Print the performance of each model
for model_name, metrics in model_performances.items():
    print(f"{model_name} Performance:")
    for metric, value in metrics.items():
        print(f" {metric}: {value:.4f}")
