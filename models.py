from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import numpy as np
import pandas as pd

# ── Color constants for consistent theming ─────────────────────────────────────
TEAL = "#2dd4bf"
PURPLE = "#a78bfa"
AMBER = "#fbbf24"
ROSE = "#fb7185"
EMERALD = "#34d399"
SURFACE = "#161b22"
BORDER = "#21262d"
TEXT_1 = "#f0f6fc"
TEXT_2 = "#8b949e"
TEXT_3 = "#484f58"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_models(X_train, y_train):
    """ Train all four classifiers and return them as a dict. """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'SVM': SVC(
            probability=True, random_state=42, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, class_weight='balanced', max_depth=5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        ),
    }
    
    logging.info("\n" + "=" * 50)
    logging.info("TRAINING MODELS")
    logging.info("=" * 50)
    logging.info(f"Input shape: {X_train.shape}")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        logging.info(f"✅ Trained {name}")
    
    return models

def get_feature_importance(model, feature_names, model_name):
    """ Extract feature importance/coefficients from trained model. """
    if model_name == 'Logistic Regression':
        if hasattr(model, 'coef_'):
            coef = model.coef_[0]
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(coef)],
                'importance': np.abs(coef),
                'coefficient': coef,
                'direction': ['increases risk' if c > 0 else 'decreases risk' for c in coef]
            }).sort_values('importance', ascending=False)
            return importance_df
    elif model_name in ['Random Forest', 'Decision Tree']:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(model.feature_importances_)],
                'importance': model.feature_importances_,
                'direction': ['influences risk'] * len(model.feature_importances_)
            }).sort_values('importance', ascending=False)
            return importance_df
    return None

def explain_prediction(model, model_name, X_sample, feature_names, feature_descriptions, user_values):
    """ Generate personalized explanation of which features influenced this prediction. """
    explanation = {
        'top_features': [],
        'recommendations': []
    }
    
    if model_name == 'Logistic Regression' and hasattr(model, 'coef_'):
        coef = model.coef_[0]
        contributions = []
        
        for i, feat in enumerate(feature_names[:len(coef)]):
            if i < len(X_sample[0]):
                contribution = coef[i] * X_sample[0][i]
                contributions.append({
                    'feature': feat,
                    'coefficient': coef[i],
                    'value': X_sample[0][i],
                    'contribution': contribution,
                    'abs_contribution': abs(contribution)
                })
        
        contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        for item in contributions[:5]:
            feature = item['feature']
            direction = "increases" if item['coefficient'] > 0 else "decreases"
            coefficient = item['coefficient']
            value = user_values.get(feature, item['value'])
            
            explanation['top_features'].append({
                'feature': feature,
                'description': feature_descriptions.get(feature, feature),
                'value': value,
                'direction': direction,
                'impact': abs(item['contribution']),
                'raw_contribution': item['contribution'],
                'coefficient': coefficient
            })
            
            # Medical knowledge-based recommendations
            if feature == 'trestbps' and value > 140:
                explanation['recommendations'].append(f"💓 **Elevated Blood Pressure** ({value:.0f} mm Hg) - Reduce salt intake, monitor BP regularly.")
            elif feature == 'chol' and value > 240:
                explanation['recommendations'].append(f"🥗 **High Cholesterol** ({value:.0f} mg/dl) - Consider diet modification and statin medication.")
            elif feature == 'fbs' and value == 1:
                explanation['recommendations'].append("🩸 **High fasting blood sugar** (>120 mg/dl) - Monitor HbA1c and manage glucose levels.")
            elif feature == 'age' and value > 60:
                explanation['recommendations'].append(f"👴 **Age {value:.0f}** - Regular cardiac checkups recommended.")
            elif feature == 'thalach' and value < 120:
                explanation['recommendations'].append(f"💪 **Low exercise capacity** (max heart rate {value:.0f}) - Consider cardiac rehabilitation.")
            elif feature == 'oldpeak' and value > 2:
                explanation['recommendations'].append(f"📉 **Significant ST depression** ({value:.1f} mm) - Stress test recommended.")
            elif feature == 'exang' and value == 1:
                explanation['recommendations'].append("🏃 **Exercise-induced angina** - Cardiology consult advised.")
            elif feature == 'sex' and value == 1:
                explanation['recommendations'].append("⚥ **Male gender** - Regular cardiac screening recommended.")
    
    elif model_name in ['Random Forest', 'Decision Tree'] and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        contributions = []
        
        for i, feat in enumerate(feature_names[:len(importances)]):
            if i < len(X_sample[0]):
                contributions.append({
                    'feature': feat,
                    'importance': importances[i],
                    'value': X_sample[0][i],
                })
        
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        for item in contributions[:5]:
            feature = item['feature']
            value = user_values.get(feature, item['value'])
            
            explanation['top_features'].append({
                'feature': feature,
                'description': feature_descriptions.get(feature, feature),
                'value': value,
                'importance': item['importance'],
                'direction': 'influences'
            })
            
            if feature == 'trestbps' and value > 140:
                explanation['recommendations'].append(f"💓 **Elevated Blood Pressure** ({value:.0f} mm Hg) is a risk factor. Reduce salt intake.")
            elif feature == 'chol' and value > 240:
                explanation['recommendations'].append(f"🥗 **High Cholesterol** ({value:.0f} mg/dl) is a risk factor. Consider diet modification.")
            elif feature == 'fbs' and value == 1:
                explanation['recommendations'].append("🩸 **High fasting blood sugar** (>120 mg/dl) indicates diabetes risk.")
            elif feature == 'age' and value > 60:
                explanation['recommendations'].append("👴 **Age > 60** increases risk. Regular cardiac checkups recommended.")
            elif feature == 'thalach' and value < 120:
                explanation['recommendations'].append(f"💪 **Low exercise capacity** - consider cardiac rehab.")
            elif feature == 'oldpeak' and value > 2:
                explanation['recommendations'].append(f"📉 **ST depression** ({value:.1f} mm) suggests ischemia.")
            elif feature == 'exang' and value == 1:
                explanation['recommendations'].append("🏃 **Exercise-induced angina** - cardiology consult advised.")
    
    # Remove duplicate recommendations
    explanation['recommendations'] = list(dict.fromkeys(explanation['recommendations']))
    
    return explanation

def cross_validate_models(X, y, n_splits=5, random_state=42):
    """ Perform Stratified K-Fold Cross-Validation for all models. """
    logging.info("\n" + "=" * 50)
    logging.info("STRATIFIED K-FOLD CROSS-VALIDATION (k=5)")
    logging.info("=" * 50)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    base_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight='balanced'
        ),
        'SVM': SVC(
            probability=True, random_state=random_state, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state, class_weight='balanced', max_depth=5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=random_state, class_weight='balanced'
        ),
    }
    
    cv_results = {}
    all_predictions = {name: {'y_true': [], 'y_pred': [], 'y_proba': []} for name in base_models.keys()}
    
    for model_name, model in base_models.items():
        logging.info(f"\n📊 Cross-validating {model_name}...")
        
        fold_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
        }
        fold_cms = []
        fold_fprs = []
        fold_tprs = []
        fold = 1
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)
            
            pca_fold = PCA(n_components=min(8, X_train_scaled.shape[1]), random_state=random_state)
            X_train_pca = pca_fold.fit_transform(X_train_scaled)
            X_val_pca = pca_fold.transform(X_val_scaled)
            
            model_clone = base_models[model_name]
            model_clone.fit(X_train_pca, y_train_fold)
            
            y_pred = model_clone.predict(X_val_pca)
            y_proba = model_clone.predict_proba(X_val_pca)[:, 1]
            
            all_predictions[model_name]['y_true'].extend(y_val_fold)
            all_predictions[model_name]['y_pred'].extend(y_pred)
            all_predictions[model_name]['y_proba'].extend(y_proba)
            
            fold_metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            fold_metrics['precision'].append(precision_score(y_val_fold, y_pred, average='binary', zero_division=0))
            fold_metrics['recall'].append(recall_score(y_val_fold, y_pred, average='binary', zero_division=0))
            fold_metrics['f1'].append(f1_score(y_val_fold, y_pred, average='binary', zero_division=0))
            
            fpr, tpr, _ = roc_curve(y_val_fold, y_proba)
            fold_metrics['roc_auc'].append(auc(fpr, tpr))
            
            fold_cms.append(confusion_matrix(y_val_fold, y_pred))
            fold_fprs.append(fpr)
            fold_tprs.append(tpr)
            
            logging.info(f"  Fold {fold}: Acc={fold_metrics['accuracy'][-1]:.3f}, "
                        f"Prec={fold_metrics['precision'][-1]:.3f}, "
                        f"Rec={fold_metrics['recall'][-1]:.3f}, "
                        f"F1={fold_metrics['f1'][-1]:.3f}, "
                        f"AUC={fold_metrics['roc_auc'][-1]:.3f}")
            fold += 1
        
        cv_results[model_name] = {
            'accuracy_mean': np.mean(fold_metrics['accuracy']),
            'accuracy_std': np.std(fold_metrics['accuracy']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'f1_mean': np.mean(fold_metrics['f1']),
            'f1_std': np.std(fold_metrics['f1']),
            'roc_auc_mean': np.mean(fold_metrics['roc_auc']),
            'roc_auc_std': np.std(fold_metrics['roc_auc']),
            'all_folds': fold_metrics,
            'confusion_matrices': fold_cms,
            'fpr_list': fold_fprs,
            'tpr_list': fold_tprs
        }
        
        logging.info(f"\n✅ {model_name} CV Results (mean ± std):")
        logging.info(f"  Accuracy: {cv_results[model_name]['accuracy_mean']:.3f} ± {cv_results[model_name]['accuracy_std']:.3f}")
        logging.info(f"  Precision: {cv_results[model_name]['precision_mean']:.3f} ± {cv_results[model_name]['precision_std']:.3f}")
        logging.info(f"  Recall: {cv_results[model_name]['recall_mean']:.3f} ± {cv_results[model_name]['recall_std']:.3f}")
        logging.info(f"  F1-Score: {cv_results[model_name]['f1_mean']:.3f} ± {cv_results[model_name]['f1_std']:.3f}")
        logging.info(f"  AUC-ROC: {cv_results[model_name]['roc_auc_mean']:.3f} ± {cv_results[model_name]['roc_auc_std']:.3f}")
    
    return cv_results, all_predictions

def evaluate_models(models, X_test, y_test):
    """ Evaluate all models on X_test / y_test. """
    results = {}
    logging.info("\n" + "=" * 50)
    logging.info("EVALUATING MODELS ON TEST SET")
    logging.info("=" * 50)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba,
        }
        
        logging.info(f"\n{name}:")
        logging.info(f"  Accuracy: {acc:.3f}")
        logging.info(f"  Precision: {prec:.3f}")
        logging.info(f"  Recall: {rec:.3f}")
        logging.info(f"  F1-Score: {f1:.3f}")
        logging.info(f"  AUC-ROC: {roc_auc:.3f}")
        logging.info(f"  Confusion Matrix:\n{cm}")
    
    return results

def plot_roc_curves_cv(cv_results):
    """Plot ROC curves for all models using cross-validation results."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [TEAL, PURPLE, AMBER, ROSE]
    
    for idx, (name, res) in enumerate(cv_results.items()):
        for fold_idx in range(len(res['fpr_list'])):
            ax.plot(
                res['fpr_list'][fold_idx],
                res['tpr_list'][fold_idx],
                linewidth=1,
                color=colors[idx % 4],
                alpha=0.2
            )
        ax.plot([], [], linewidth=2.5, color=colors[idx % 4], 
                label=f"{name} (AUC={res['roc_auc_mean']:.3f} ± {res['roc_auc_std']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - 5-Fold Cross-Validation', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    
    return fig

def plot_confusion_matrix_cv(cv_results, model_name):
    """Plot average confusion matrix from cross-validation."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cms = cv_results[model_name]['confusion_matrices']
    avg_cm = np.mean(cms, axis=0).astype(int)
    cm_percent = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.empty_like(avg_cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{avg_cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(
        avg_cm, annot=annot, fmt='', cmap='Blues',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease'],
        cbar=True, square=True, ax=ax,
    )
    ax.set_title(f'{model_name} (Average over 5 folds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    fig.tight_layout()
    
    return fig

def save_model(model, filename):
    joblib.dump(model, filename)
    logging.info(f"💾 Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    logging.info(f"📂 Model loaded from {filename}")
    return model