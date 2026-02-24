"""
Evaluator Module

This module provides the Evaluator class for model training, prediction, and evaluation
of fairness and accuracy metrics. The class supports various classification models and
provides methods to calculate bias concentration (epsilon), fairness metrics, and
accuracy metrics.

Key features:
- Model initialization based on configuration parameters
- Support for different types of classifiers (scikit-learn, XGBoost, LightGBM, etc.)
- Training strategy with full dataset for first run and sampling for subsequent runs
- Calculation of various fairness metrics (BNC, BPC, CUAE, EOpp, etc.)
- Calculation of accuracy metrics (ACC, F1, Recall, Precision)
- Calculation of bias concentration metric (epsilon)

Usage:
    evaluator = Evaluator(label_O='SEX', label_Y='default payment next month', 
                          cate_attrs=[], num_attrs=[])
    metrics = evaluator.evaluate(X_train, Y_train, O_train, X_test, Y_test, O_test)
    epsilon = evaluator.calculate_epsilon(X, Y, O, num_attrs, cate_attrs)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances

# MODELS
# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost
# Import configuration parameters
from core_config import PARAMS_EVAL_H_ORDER, PARAMS_EVAL_DIST_METRIC, PARAMS_MAIN_TRAINING_RATE, SEED, PARAMS_MAIN_CLASSIFIER, VERBOSE, PARAMS_EVAL_METRIC_ACCURACY, PARAMS_EVAL_SCALE, PARAMS_EVAL_METRIC_FAIRNESS, PARAMS_EVAL_CAT, PARAMS_EVAL_NUM, PARAMS_EVAL_SUM

import warnings
warnings.filterwarnings("ignore")


class Evaluator:
    """
    Evaluator class for model training, prediction, and evaluation of metrics.
    
    This class provides functionality to train classification models, make predictions,
    and calculate various fairness and accuracy metrics. It also includes methods to
    compute bias concentration (epsilon) to measure the distribution of bias across features.
    """
    
    def __init__(self, label_O='SEX', label_Y='default payment next month', cate_attrs=[], num_attrs=[]):
        """
        Initialize the Evaluator class.
        
        Args:
            label_O (str or list): Protected attribute(s)
            label_Y (str): Target attribute
            cate_attrs (list): List of categorical attribute names
            num_attrs (list): List of numerical attribute names
        """
        # Initialize evaluator and model based on configuration
        self.model = self._create_model()
        self.first_train = True
        self.random_state = SEED
        self.label_O = label_O
        self.label_Y = label_Y
        self.results_df = None  # To store results dataframe
        self.metrics_results = {}  # To store calculated metrics
        self.h_order = PARAMS_EVAL_H_ORDER
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        
        
    def _create_model(self):
        """
        Create model based on PARAMS_MAIN_CLASSIFIER from config.
        
        Returns:
            model: Initialized classification model
        """
        from core_config import PARAMS_MAIN_CLASSIFIER
        if VERBOSE:
            print(f"Using classifier: {PARAMS_MAIN_CLASSIFIER}")

        if PARAMS_MAIN_CLASSIFIER == 'LR':
            # StandardScaler is now handled by Transform class, not here
            return LogisticRegression(random_state=SEED, max_iter=1000)
        elif PARAMS_MAIN_CLASSIFIER == 'DT':
            return DecisionTreeClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'KNN':
            return KNeighborsClassifier()
        elif PARAMS_MAIN_CLASSIFIER == 'GBDT':
            return GradientBoostingClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'ADABoost':
            return AdaBoostClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'NB':
            return GaussianNB()
        elif PARAMS_MAIN_CLASSIFIER == 'SVM':
            return SVC(random_state=SEED, max_iter=100, probability=True)
        elif PARAMS_MAIN_CLASSIFIER == 'MLP':
            return MLPClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'XGBoost':
            return xgb.XGBClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'RF':
            return RandomForestClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'LGBM':
            return lgb.LGBMClassifier(random_state=SEED, force_col_wise=True, verbose=-1)
        elif PARAMS_MAIN_CLASSIFIER == 'CatBoost':
            return catboost.CatBoostClassifier(random_state=SEED, verbose=0)
        elif PARAMS_MAIN_CLASSIFIER == 'LDA':
            return LinearDiscriminantAnalysis()
        elif PARAMS_MAIN_CLASSIFIER == 'QDA':
            return QuadraticDiscriminantAnalysis(reg_param=0.01)
        else:
            raise ValueError(f"Unsupported classifier type: {PARAMS_MAIN_CLASSIFIER}")

    def fit(self, X, Y, O):
        """
        Train the model using separate X, Y, O data format.
        - First training always uses the full dataset
        - Subsequent trainings use a sample according to PARAMS_MAIN_TRAINING_RATE
        
        Args:
            X (pd.DataFrame): Feature data
            Y (pd.Series or array-like): Target variable
            O (pd.DataFrame): Protected attributes
        """
        # Check if O is a DataFrame
        if not isinstance(O, pd.DataFrame):
            O = pd.DataFrame(O)
        
        # For other models
        # First training: full dataset
        if self.first_train:
            if VERBOSE:
                import sys
                print("First training, using full dataset", flush=True, file=sys.stderr)
                print(f"[DEBUG] Training data - X shape: {X.shape}, Y shape: {Y.shape}", flush=True, file=sys.stderr)
                print(f"[DEBUG] Y value counts: {Y.value_counts().to_dict()}", flush=True, file=sys.stderr)
                print(f"[DEBUG] X columns: {X.columns.tolist()}", flush=True, file=sys.stderr)
                print(f"[DEBUG] X first row: {X.iloc[0].to_dict()}", flush=True, file=sys.stderr)
            
            self.model.fit(X, Y)
            
            if VERBOSE:
                import sys
                # Test prediction on training data
                y_train_pred = self.model.predict(X[:100])  # Test on first 100 samples
                print(f"[DEBUG] After training - prediction on first 100 training samples: {pd.Series(y_train_pred).value_counts().to_dict()}", flush=True, file=sys.stderr)
                
                # Check if model is Pipeline (with StandardScaler) or direct classifier
                if hasattr(self.model, 'named_steps'):
                    # Pipeline with StandardScaler
                    actual_model = self.model.named_steps['classifier']
                    print(f"[DEBUG] Using StandardScaler pipeline", flush=True, file=sys.stderr)
                else:
                    actual_model = self.model
                    print(f"[DEBUG] Using raw model (no scaling)", flush=True, file=sys.stderr)
                
                if hasattr(actual_model, 'coef_'):
                    print(f"[DEBUG] Model coefficients shape: {actual_model.coef_.shape}", flush=True, file=sys.stderr)
                    print(f"[DEBUG] Model intercept: {actual_model.intercept_}", flush=True, file=sys.stderr)
                    coef_abs_mean = np.abs(actual_model.coef_).mean()
                    coef_abs_max = np.abs(actual_model.coef_).max()
                    coef_abs_min = np.abs(actual_model.coef_).min()
                    print(f"[DEBUG] Coefficient stats - mean: {coef_abs_mean:.6f}, max: {coef_abs_max:.6f}, min: {coef_abs_min:.6f}", flush=True, file=sys.stderr)
            
            self.first_train = False
        else:
            # Subsequent training: sample according to PARAMS_MAIN_TRAINING_RATE
            if 0 < PARAMS_MAIN_TRAINING_RATE < 1:
                if VERBOSE:
                    print(f"Subsequent training, sampling at {PARAMS_MAIN_TRAINING_RATE} ratio")
                # Sample indices
                sample_indices = X.sample(frac=PARAMS_MAIN_TRAINING_RATE, random_state=SEED).index
                # Get sampled data
                X_train = X.loc[sample_indices]
                if isinstance(Y, pd.Series):
                    Y_train = Y.loc[sample_indices]
                else:
                    # If Y is not a Series, assume it's a numpy array with the same index as X
                    Y_train = Y[sample_indices]
                
                self.model.fit(X_train, Y_train)
            else:
                if VERBOSE:
                    print("Subsequent training, using full dataset")
                self.model.fit(X, Y)

    def predict(self, X_test, Y_test, O_test):
        """
        Make predictions on test data using separate X, Y, O data format.
        
        Args:
            X_test (pd.DataFrame): Feature data
            Y_test (pd.Series or array-like): Target variable
            O_test (pd.DataFrame): Protected attributes
        """
        # Normalize label_O to a list
        self.label_O = self.label_O if isinstance(self.label_O, list) else [self.label_O]
        
        # For PyTorch Tabular models, create a single DataFrame
        if PARAMS_MAIN_CLASSIFIER in ['TabNet', 'TabTransformer', 'CategoryEmbedding', 'GATE', 
                                     'FTTransformer', 'AutoInt', 'DANet', 'GANDALF', 'NODE'] and PYTORCH_TABULAR_AVAILABLE:
            # Create a single DataFrame with X, Y, O
            df = pd.concat([X_test.reset_index(drop=True), O_test.reset_index(drop=True)], axis=1)
            # Add Y to the DataFrame
            if isinstance(Y_test, pd.Series):
                df[self.label_Y] = Y_test.reset_index(drop=True)
            else:
                df[self.label_Y] = Y_test
            
            # Make predictions
            pred_df = self.model.predict(df)
            pred_col = f'{self.label_Y}_prediction'
            prob_col = f'{self.label_Y}_1_probability'
            if pred_col not in pred_df.columns or prob_col not in pred_df.columns:
                raise RuntimeError(
                    f"Unexpected TabularModel predict output; missing {pred_col} or {prob_col}"
                )
            y_pred = pred_df[pred_col].values
            score_S = pred_df[prob_col].values
        else:
            # For other models
            # Make predictions
            if hasattr(self.model, "predict_proba"):
                score_S = self.model.predict_proba(X_test)[:, 1]
            else:
                print("Warning: model has no predict_proba, using hard predictions as score.")
                score_S = self.model.predict(X_test).astype(float)
            y_pred = self.model.predict(X_test)
        
        # Build results_df
        results_data = {
            'label_Y': Y_test,
            'pred_Y': y_pred,
            'score_S': score_S,
        }
        
        # Add protected attributes
        if isinstance(O_test, pd.DataFrame):
            for col in O_test.columns:
                results_data[col] = O_test[col]
        else:
            # If O_test is not a DataFrame, assume it's a single protected attribute
            if isinstance(self.label_O, list) and len(self.label_O) > 0:
                results_data[self.label_O[0]] = O_test
            else:
                results_data[self.label_O] = O_test
        
        self.results_df = pd.DataFrame(results_data)
        
        if VERBOSE:
            import sys
            pred_dist = self.results_df['pred_Y'].value_counts().to_dict()
            print(f"[DEBUG] Prediction distribution: {pred_dist}", flush=True, file=sys.stderr)
    
    @staticmethod
    def _confusion_counts(df):
        """
        Calculate confusion matrix statistics.
        
        Args:
            df (pd.DataFrame): Results dataframe with 'pred_Y' and 'label_Y' columns
            
        Returns:
            dict: Dictionary with confusion matrix statistics
        """
        y_pred = df['pred_Y'].astype(bool)
        y_true = df['label_Y'].astype(bool)
        
        # Confusion matrix
        tp = (y_pred & y_true).sum()   # True Positive
        fp = (y_pred & ~y_true).sum()  # False Positive
        fn = (~y_pred & y_true).sum()  # False Negative
        tn = (~y_pred & ~y_true).sum() # True Negative
        
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        FNR = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        
        PPV = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        NPV = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        FDR = fp / (tp + fp) if (tp + fp) > 0 else 0.0  # False Discovery Rate (= 1 - PPV)
        FOR = fn / (fn + tn) if (fn + tn) > 0 else 0.0  # False Omission Rate (= 1 - NPV)
        
        ACC = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0  # Accuracy
        PPOS = y_pred.mean()              # Positive Prediction Rate
        
        return dict(
            tp=tp, fp=fp, fn=fn, tn=tn,
            TPR=TPR, FPR=FPR, FNR=FNR, TNR=TNR,
            PPV=PPV, NPV=NPV, ACC=ACC, PPOS=PPOS,
            FDR=FDR, FOR=FOR
        )
    
    def _calculate_bpc(self, label_O, group_pairs):
        """
        Calculate Between Positive Classes (BPC) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: BPC value
        """
        vals = []
        for group_a, group_b in group_pairs:
            group_a_pos = self.results_df[(self.results_df[label_O] == group_a) &
                                          (self.results_df['label_Y'] == 1)]
            group_b_pos = self.results_df[(self.results_df[label_O] == group_b) &
                                          (self.results_df['label_Y'] == 1)]
            m_a = group_a_pos['score_S'].mean() if len(group_a_pos) else 0.0
            m_b = group_b_pos['score_S'].mean() if len(group_b_pos) else 0.0
            vals.append(abs(m_a - m_b))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_bnc(self, label_O, group_pairs):
        """
        Calculate Between Negative Classes (BNC) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: BNC value
        """
        vals = []
        for group_a, group_b in group_pairs:
            group_a_neg = self.results_df[(self.results_df[label_O] == group_a) &
                                          (self.results_df['label_Y'] == 0)]
            group_b_neg = self.results_df[(self.results_df[label_O] == group_b) &
                                          (self.results_df['label_Y'] == 0)]
            m_a = group_a_neg['score_S'].mean() if len(group_a_neg) else 0.0
            m_b = group_b_neg['score_S'].mean() if len(group_b_neg) else 0.0
            vals.append(abs(m_a - m_b))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_cuae(self, label_O, group_pairs):
        """
        Calculate Conditional Use Accuracy Equality (CUAE) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: CUAE value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            # max(|ΔPPV|, |ΔNPV|)
            vals.append(max(abs(stats_a['PPV'] - stats_b['PPV']),
                            abs(stats_a['NPV'] - stats_b['NPV'])))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_eopp(self, label_O, group_pairs):
        """
        Calculate Equal Opportunity (EOpp) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: EOpp value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            tpr_diff = abs(stats_a['TPR'] - stats_b['TPR'])
            vals.append(tpr_diff)
            if VERBOSE and tpr_diff == 0:
                import sys
                print(f"[DEBUG] EOpp=0 for {label_O}: group {group_a} TPR={stats_a['TPR']:.6f}, group {group_b} TPR={stats_b['TPR']:.6f}", 
                      flush=True, file=sys.stderr)
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_eo(self, label_O, group_pairs):
        """
        Calculate Equalized Odds (EO) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: EO value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            # max(|ΔTPR|, |ΔFPR|)
            vals.append(max(abs(stats_a['TPR'] - stats_b['TPR']),
                            abs(stats_a['FPR'] - stats_b['FPR'])))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_fdrp(self, label_O, group_pairs):
        """
        Calculate False Discovery Rate Parity (FDRP) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: FDRP value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['FDR'] - stats_b['FDR']))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_forp(self, label_O, group_pairs):
        """
        Calculate False Omission Rate Parity (FORP) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: FORP value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['FOR'] - stats_b['FOR']))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_fnrb(self, label_O, group_pairs):
        """
        Calculate False Negative Rate Balance (FNRB) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: FNRB value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['FNR'] - stats_b['FNR']))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_fprb(self, label_O, group_pairs):
        """
        Calculate False Positive Rate Balance (FPRB) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: FPRB value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['FPR'] - stats_b['FPR']))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_npvp(self, label_O, group_pairs):
        """
        Calculate Negative Predictive Value Parity (NPVP) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: NPVP value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['NPV'] - stats_b['NPV']))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_oae(self, label_O, group_pairs):
        """
        Calculate Overall Accuracy Equality (OAE) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: OAE value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['ACC'] - stats_b['ACC']))
        return sum(vals) / len(vals) if vals else 0.0
        
    def _calculate_ppvp(self, label_O, group_pairs):
        """
        Calculate Positive Predictive Value Parity (PPVP) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: PPVP value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            vals.append(abs(stats_a['PPV'] - stats_b['PPV']))
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_sp(self, label_O, group_pairs):
        """
        Calculate Statistical Parity (SP) fairness metric.
        
        Args:
            label_O (str): Protected attribute name
            group_pairs (list): List of group pairs
            
        Returns:
            float: SP value
        """
        vals = []
        for group_a, group_b in group_pairs:
            stats_a = self._confusion_counts(self.results_df[self.results_df[label_O] == group_a])
            stats_b = self._confusion_counts(self.results_df[self.results_df[label_O] == group_b])
            ppos_diff = abs(stats_a['PPOS'] - stats_b['PPOS'])
            vals.append(ppos_diff)
            if VERBOSE and ppos_diff == 0:
                import sys
                print(f"[DEBUG] SP=0 for {label_O}: group {group_a} PPOS={stats_a['PPOS']:.6f}, group {group_b} PPOS={stats_b['PPOS']:.6f}", 
                      flush=True, file=sys.stderr)
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_acc(self, confusion_stats=None):
        """
        Calculate overall Accuracy (ACC) metric.
        
        Args:
            confusion_stats (dict, optional): Precomputed confusion matrix statistics
        
        Returns:
            float: ACC value
        """
        if confusion_stats is None:
            confusion_stats = self._confusion_counts(self.results_df)
        return confusion_stats['ACC']
    
    def _calculate_f1(self, confusion_stats=None):
        """
        Calculate overall F1 Score (F1) metric.
        
        Args:
            confusion_stats (dict, optional): Precomputed confusion matrix statistics
        
        Returns:
            float: F1 value
        """
        if confusion_stats is None:
            confusion_stats = self._confusion_counts(self.results_df)
        precision = confusion_stats['PPV']
        recall = confusion_stats['TPR']
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return f1
    
    def _calculate_recall(self, confusion_stats=None):
        """
        Calculate overall Recall metric.
        
        Args:
            confusion_stats (dict, optional): Precomputed confusion matrix statistics
        
        Returns:
            float: Recall value
        """
        if confusion_stats is None:
            confusion_stats = self._confusion_counts(self.results_df)
        return confusion_stats['TPR']
    
    def _calculate_precision(self, confusion_stats=None):
        """
        Calculate overall Precision metric.
        
        Args:
            confusion_stats (dict, optional): Precomputed confusion matrix statistics
        
        Returns:
            float: Precision value
        """
        if confusion_stats is None:
            confusion_stats = self._confusion_counts(self.results_df)
        return confusion_stats['PPV']
    
    def calculate_metrics(self):
        """
        Calculate all requested fairness and accuracy metrics.
        
        Returns:
            dict: Dictionary with calculated metrics
        """
        # Check if results_df exists
        if self.results_df is None:
            raise ValueError("No results available. Run predict first.")
        
        # Extract data from results_df
        exclude_cols = {'label_Y', 'pred_Y', 'score_S'}
        label_O_columns = [c for c in self.results_df.columns if c not in exclude_cols]
        
        # Calculate classifier metrics using encapsulated methods
        classifier_metrics = {}
        
        # Calculate confusion matrix once and reuse it for all accuracy metrics
        confusion_stats = self._confusion_counts(self.results_df)
        
        # Create a map for accuracy metrics, similar to fairness metrics
        accuracy_metric_map = {
            'ACC': self._calculate_acc,
            'F1': self._calculate_f1,
            'Recall': self._calculate_recall,
            'Precision': self._calculate_precision
        }
        
        # Calculate each accuracy metric using the precomputed confusion_stats
        for metric_name in PARAMS_EVAL_METRIC_ACCURACY:
            if metric_name in accuracy_metric_map:
                classifier_metrics[metric_name] = accuracy_metric_map[metric_name](confusion_stats)
        
        # Calculate fairness metrics for each label_O
        metric_map = {
            'BNC'  : self._calculate_bnc,
            'BPC'  : self._calculate_bpc,
            'CUAE' : self._calculate_cuae,
            'EOpp' : self._calculate_eopp,
            'EO'   : self._calculate_eo,
            'FDRP' : self._calculate_fdrp,
            'FORP' : self._calculate_forp,
            'FNRB' : self._calculate_fnrb,
            'FPRB' : self._calculate_fprb,
            'NPVP' : self._calculate_npvp,
            'OAE'  : self._calculate_oae,
            'PPVP' : self._calculate_ppvp,
            'SP'   : self._calculate_sp
        }
        
        fairness_metrics = {name: {} for name in PARAMS_EVAL_METRIC_FAIRNESS if name in metric_map}
        
        for label_O in label_O_columns:
            groups = self.results_df[label_O].unique()
            if len(groups) < 2:
                print(f"Warning: Only one group found in {label_O}. Cannot calculate fairness metrics.")
                continue
            
            group_pairs = list(combinations(groups, 2))
            
            for name in fairness_metrics.keys():
                func = metric_map[name]
                fairness_metrics[name][label_O] = func(label_O, group_pairs)
        
        # Combine all metrics
        self.metrics_results = {**classifier_metrics, **fairness_metrics}
        
        if VERBOSE:
            import sys
            print("Metrics calculated successfully", flush=True, file=sys.stderr)
            print(f"[DEBUG] Fairness metrics structure:", flush=True, file=sys.stderr)
            for metric_name, metric_value in fairness_metrics.items():
                print(f"  {metric_name}: {metric_value}", flush=True, file=sys.stderr)
        
        return self.metrics_results
    
    def evaluate(self, X_train, Y_train, O_train, X_test, Y_test, O_test):
        """
        Main evaluation pipeline using separate X, Y, O data format.
        
        Args:
            X_train (pd.DataFrame): Feature data for training
            Y_train (pd.Series or array-like): Target variable for training
            O_train (pd.DataFrame): Protected attributes for training
            X_test (pd.DataFrame): Feature data for testing
            Y_test (pd.Series or array-like): Target variable for testing
            O_test (pd.DataFrame): Protected attributes for testing
            
        Returns:
            dict: Dictionary with calculated metrics
        """
        # Train the model using separate data
        self.fit(X_train, Y_train, O_train)
        
        # Predict on test data using separate data
        self.predict(X_test, Y_test, O_test)
        
        # Calculate metrics
        return self.calculate_metrics()
    
    def calculate_epsilon(self, X, O, cate_attrs, num_attrs):
        """
        Calculate epsilon metric for measuring bias concentration using separate X, Y, O data format.
        
        Args:
            X (pd.DataFrame): Feature data
            Y (pd.Series or array-like): Target variable
            O (pd.DataFrame): Protected attributes
            cate_attrs (list): List of categorical attribute names
            num_attrs (list): List of numerical attribute names
            
        Returns:
            dict: Dictionary with epsilon values for each protected attribute
        """
        # Create a single DataFrame from X, Y, O
        df_data = pd.concat([X.reset_index(drop=True), O.reset_index(drop=True)], axis=1)
        
        
        # Handle both single and multiple protected attributes
        label_O_list = [self.label_O] if isinstance(self.label_O, str) else self.label_O
        epsilon_results = {}
        
        # Set default h_order if not specified
        if self.h_order == 'default':
            self.h_order = len(X.columns) - 1
        
        # Create a copy of df_data with all label_O columns dropped
        all_label_O = label_O_list
        df_data_no_O = df_data.drop(all_label_O, axis=1) if len(all_label_O) > 0 else df_data.copy()
        
        # Calculate epsilon for each protected attribute
        for label_O in label_O_list:
            # Get unique values and their combinations
            unique_values = df_data[label_O].unique()
            comb_label_arr = combinations(unique_values, 2)
            df_S_full = pd.DataFrame()
            
            # Process each combination of protected attribute values
            for p, n in comb_label_arr:
                # Filter data for current combination
                mask = df_data[label_O].isin([p, n])
                df_filtered = df_data_no_O[mask].copy()
                # Add back the current label_O for filtering
                df_filtered[label_O] = df_data[label_O][mask]
                
                # Normalize numerical attributes
                for col in num_attrs:
                    if col in df_filtered.columns:
                        min_val, max_val = df_filtered[col].min(), df_filtered[col].max()
                        if min_val != max_val:
                            df_filtered[col] = (df_filtered[col] - min_val) / (max_val - min_val)
                
                # Calculate differences for categorical attributes
                cat_diff = {}
                for attr in cate_attrs:
                    if attr in [label_O, self.label_Y] or attr not in df_filtered.columns:
                        continue
                    
                    p_data = df_filtered[df_filtered[label_O] == p][attr]
                    n_data = df_filtered[df_filtered[label_O] == n][attr]
                    
                    # Calculate counts
                    p_counts = p_data.value_counts()
                    n_counts = n_data.value_counts()
                    N_c1 = p_counts.sum()
                    N_c2 = n_counts.sum()
                    
                    # Combine categories to ensure consistent indexing
                    combined = p_counts.reindex(pd.unique([*p_counts.index, *n_counts.index]), fill_value=0)
                    n_counts = n_counts.reindex(combined.index, fill_value=0)
                    
                    if PARAMS_EVAL_CAT == 'cat-a':
                        # Calculate categorical difference (using centroid method)
                        K = len(combined)
                        proportions_p = combined / N_c1
                        proportions_n = n_counts / N_c2
                        # Formula 3a: (1/K) * Σ |N_k^c1/N^c1 - N_k^c2/N^c2|
                        cat_diff[attr] = (1 / K) * (proportions_p - proportions_n).abs().sum()
                    elif PARAMS_EVAL_CAT == 'cat-b':
                        # For chi-square method
                        # Formula 3b: (1/2) * Σ (N_k^c1 - N_k^c2)² / (N_k^c1 + N_k^c2)
                        numerator = (combined - n_counts) ** 2
                        denominator = combined + n_counts
                        denominator[denominator == 0] = 1  # Avoid division by zero
                        cat_diff[attr] = 0.5 * (numerator / denominator).sum()
                
                # Calculate differences for numerical attributes
                if PARAMS_EVAL_NUM != 'num-d':
                    num_diff = pd.Series(dtype='float64')
                    for col in num_attrs:
                        # Get values for both groups
                        p_vals = df_filtered[df_filtered[label_O] == p][col].values
                        n_vals = df_filtered[df_filtered[label_O] == n][col].values
                        
                        if PARAMS_EVAL_NUM == 'num-a':
                            # Method 1: Centroid difference (absolute mean difference)
                            # Formula 5a: |X̄^c1 - X̄^c2|
                            p_mean = p_vals.mean()
                            n_mean = n_vals.mean()
                            num_diff[col] = abs(p_mean - n_mean)
                        elif PARAMS_EVAL_NUM == 'num-b':
                            # Method 2: Distributional difference (KS test)
                            from scipy.stats import ks_2samp
                            ks_stat, _ = ks_2samp(p_vals, n_vals)
                            num_diff[col] = ks_stat
                        elif PARAMS_EVAL_NUM == 'num-c':
                            # Method 3: Pairwise difference
                            if len(p_vals) > 0 and len(n_vals) > 0:
                                # Create matrix of absolute differences using broadcasting
                                # Shape: (len(p_vals), len(n_vals))
                                diff_matrix = np.abs(p_vals[:, np.newaxis] - n_vals)
                                # Calculate sum of all elements and apply formula
                                total = diff_matrix.sum()
                                num_diff[col] = 0.5 * total
                        
                    # Scale differences
                    if PARAMS_EVAL_SCALE == 'min':
                        if not num_diff.empty:
                            num_diff = num_diff / num_diff.min()
                        if cat_diff:
                            cat_diff_series = pd.Series(cat_diff)
                            cat_diff = (cat_diff_series / cat_diff_series.min()).to_dict()
                    elif PARAMS_EVAL_SCALE == 'mean':
                        if not num_diff.empty:
                            num_diff = num_diff / num_diff.mean()
                        if cat_diff:
                            cat_diff_series = pd.Series(cat_diff)
                            cat_diff = (cat_diff_series / cat_diff_series.mean()).to_dict()
                    elif PARAMS_EVAL_SCALE == 'sigma':
                        if not num_diff.empty:
                            num_diff = (num_diff - num_diff.mean()) / num_diff.std()
                            num_diff = num_diff - num_diff.min()
                        if cat_diff:
                            cat_diff_series = pd.Series(cat_diff)
                            cat_diff_series = (cat_diff_series - cat_diff_series.mean()) / cat_diff_series.std()
                            cat_diff_series = cat_diff_series - cat_diff_series.min()
                            cat_diff = cat_diff_series.to_dict()
                    
                    # Combine results for this combination
                    comb_key = f"{p}_{n}"
                    df_S_full[comb_key] = pd.concat([num_diff, pd.Series(cat_diff)])
                elif PARAMS_EVAL_NUM == 'num-d':
                    comb_key = f"{p}_{n}"
                    df_S_full[comb_key] = pd.Series(cat_diff)
                
            # Prepare index array for distance matrix
            index_arr = list(df_S_full.index) + ['origin']
            distance_matrix = pd.DataFrame(np.nan, index=index_arr, columns=index_arr)
            np.fill_diagonal(distance_matrix.values, 0)  # Set diagonal to 0
            # Calculate squared differences for faster computation
            df_S_sq = df_S_full ** 2
            temp_dict = {}
            
            # Get all subsets of attributes based on h_order
            def get_subsets(indexes, h_order):
                h_order = max(h_order, -1)
                subsets = []
                for i in range(len(indexes), h_order, -1):
                    subsets.extend(combinations(indexes, i))
                return [list(sub) for sub in subsets]
            
            sub_indexes = get_subsets(X.columns, self.h_order - 1)

            # Calculate distances based on selected method
            if PARAMS_EVAL_NUM != 'num-d':
                if PARAMS_EVAL_SUM == 'd1A':
                    for sub in sub_indexes:
                        temp_dict[tuple(sorted(sub))] = df_S_sq.loc[sub].mean() ** 0.5
                elif PARAMS_EVAL_SUM == 'd1B':
                    for sub in sub_indexes:
                        temp_dict[tuple(sorted(sub))] = df_S_sq.loc[sub].sum() ** 0.5
            elif PARAMS_EVAL_NUM == 'num-d':
                for sub in sub_indexes:
                    cate_sub = [s for s in sub if s in cate_attrs]
                    cate_temp = df_S_sq.loc[cate_sub].mean() ** 0.5
                    
                    num_sub = [s for s in sub if s in num_attrs]
                    num_temp = 0
                    if num_sub:  # 存在数值子属性时计算
                        X_num = df_data[num_sub].values.astype(np.float32)
                        labels = df_data[self.label_O].values

                        dist_matrix = pairwise_distances(X_num, metric=PARAMS_EVAL_DIST_METRIC)
                        print(dist_matrix)

                        if PARAMS_EVAL_SCALE == 'mean':
                            dist_matrix = dist_matrix / np.mean(dist_matrix)
                        elif PARAMS_EVAL_SCALE == 'min':
                            dist_matrix = dist_matrix / np.mean(dist_matrix, axis=1).min()
                        elif PARAMS_EVAL_SCALE == 'zscore':
                            dist_matrix = (dist_matrix - dist_matrix.mean()) / dist_matrix.std()
                        
                        delta_matrix = (labels[:, np.newaxis] == labels).astype(np.int8)

                        mask = np.triu(np.ones_like(dist_matrix, dtype=np.bool_), k=1)

                        num_temp = (dist_matrix * (2 * delta_matrix - 1) * mask).mean()
                    temp_dict[tuple(sorted(sub))] = cate_temp + num_temp

            # Define distance calculation function
            def calculate_distance(attr_1, attr_2):
                # Get available indexes excluding the two attributes
                available_indexes = X.columns.tolist() + ['origin']
                if attr_1 != 'origin':
                    available_indexes.remove(attr_1)
                if attr_2 != 'origin':
                    available_indexes.remove(attr_2)
                
                # Adjust h_order based on remaining attributes
                adjusted_h_order = self.h_order-(len(X.columns.tolist() + ['origin']) - len(available_indexes))
                temp_index_list = get_subsets(available_indexes, adjusted_h_order)
        
                result_list = []
                for s in temp_index_list:
                    s1 = sorted(s + [attr_1]) if attr_1 != 'origin' else sorted(s)
                    s2 = sorted(s + [attr_2]) if attr_2 != 'origin' else sorted(s)
                    
                    val1 = temp_dict.get(tuple(s1), 0)
                    val2 = temp_dict.get(tuple(s2), 0)
                    result_list.append(val1 - val2)
                
                # Ensure all elements in result_list are scalars
                scalar_results = []
                for item in result_list:
                    if hasattr(item, 'shape') and len(item.shape) > 0:
                        # If item is an array, take its mean value
                        scalar_results.append(float(np.mean(item)))
                    else:
                        scalar_results.append(float(item))
                
                return np.mean(np.abs(scalar_results)) if scalar_results else 0
            
            # Fill distance matrix
            # Set all diagonal elements to 0
            for attr in X.columns:
                distance_matrix.loc[attr, attr] = 0
            
            # Fill non-diagonal elements
            for attr_1, attr_2 in combinations(X.columns, 2):
                dist = calculate_distance(attr_1, attr_2)
                distance_matrix.loc[attr_1, attr_2] = dist
                distance_matrix.loc[attr_2, attr_1] = dist
            
            # Set all origin row and column values to 0
            if 'origin' in distance_matrix.index:
                distance_matrix.loc['origin', :] = 0
                distance_matrix.loc[:, 'origin'] = 0
            
            # Remove 'origin' row and column before applying MDS to handle NaN values
            if 'origin' in distance_matrix.index:
                mds_matrix = distance_matrix.drop('origin', axis=0).drop('origin', axis=1)
            else:
                mds_matrix = distance_matrix.copy()
            
            # Handle any remaining NaN values
            mds_matrix = mds_matrix.fillna(0)
            
            # Apply MDS to get epsilon values
            try:
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED,
                          n_init=4, max_iter=10000, eps=1e-10)
                pts = mds.fit_transform(mds_matrix)
                
                # Add back the origin point at (0,0)
                origin_point = np.zeros((1, 2))
                pts_with_origin = np.vstack((pts, origin_point))
                pts_with_origin = pts_with_origin - pts_with_origin[-1]  # Center around origin
                
                # Calculate epsilon as Euclidean distance from origin
                # Ensure index matches the original distance_matrix
                if 'origin' in distance_matrix.index:
                    # Create new index without 'origin' for the MDS points
                    non_origin_index = [idx for idx in distance_matrix.index if idx != 'origin']
                    # Create DataFrame with non-origin points first, then origin
                    epsilon_concentration = pd.DataFrame(pts_with_origin, index=non_origin_index + ['origin']).T
                else:
                    epsilon_concentration = pd.DataFrame(pts, index=distance_matrix.index).T
                df_epsilon = np.sqrt((epsilon_concentration ** 2).sum())
                
                # Sort and store results
                epsilon_results[label_O] = df_epsilon.sort_values(ascending=False)
            except Exception as e:
                print(f"Error calculating MDS for {label_O}: {e}")
                epsilon_results[label_O] = pd.Series({attr: 0 for attr in cleaned_attrs})
        
        return epsilon_results

