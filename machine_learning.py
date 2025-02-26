# ------------------------------------------
# Auxiliares

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict
import openpyxl


# Pré-processamento e Transformações
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, KBinsDiscretizer, RobustScaler
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Modelos e camadas de Keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# ------------------------------------------
# Métricas de Classificação
# ------------------------------------------
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ------------------------------------------
# Métricas de Regressão
# ------------------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr

# ------------------------------------------
# Classificação
# ------------------------------------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------
# Regressão
# ------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


import numpy as np
import pandas as pd

def build_dnn(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_cnn(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape, 1)))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_negbin_dnn(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate_with_cv(model, X_train, y_train, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)


        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]


        model.fit(X_fold_train, y_fold_train)

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_fold_val)
            y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
            auc_scores.append(roc_auc_score(y_fold_val, y_pred_proba[:, 1]))
        else:
            y_pred = model.predict(X_fold_val)
            auc_scores.append(roc_auc_score(y_fold_val, y_pred))


        accuracies.append(accuracy_score(y_fold_val, y_pred))
        f1_scores.append(f1_score(y_fold_val, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_fold_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_fold_val, y_pred, zero_division=0))

    return {
        'accuracy': np.mean(accuracies),
        'f1_score': np.mean(f1_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'auc': np.mean(auc_scores)
    }

def main_classification(class_community, sampling_strategy):
    if class_community['type'] == 'Classification':
        print("Aplicando modelos...")

        X = class_community['data'].iloc[:, :-1]
        y = class_community['data'].iloc[:, -1].replace({'positive': 1.0, 'negative': 0.0})

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

        transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ]
        )
        X_transformed = transformer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

        # Aplicando SMOTE, se necessário
        if sampling_strategy is not None:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=2)
            X_train, y_train = smote.fit_resample(X_train, y_train)


        results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'AUC'])

        # 1. DNN
        dnn_model = KerasClassifier(
            model=build_dnn,
            input_shape=X_train.shape[1],
            epochs=10,
            batch_size=32,
            verbose=0,
        )
        dnn_results = train_and_evaluate_with_cv(dnn_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'DNN',
                                                           'Accuracy': dnn_results['accuracy'],
                                                           'F1_Score': dnn_results['f1_score'],
                                                           'Precision': dnn_results['precision'],
                                                           'Recall': dnn_results['recall'],
                                                           'AUC': dnn_results['auc']
                                                           }])], ignore_index=True)

        # 2. CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        cnn_model = KerasClassifier(
            model=build_cnn,
            input_shape=X_train.shape[1],
            epochs=10,
            batch_size=32,
            verbose=0
        )
        cnn_results = train_and_evaluate_with_cv(cnn_model, X_train_cnn, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'CNN',
                                                           'Accuracy': cnn_results['accuracy'],
                                                           'F1_Score': cnn_results['f1_score'],
                                                           'Precision': cnn_results['precision'],
                                                           'Recall': cnn_results['recall'],
                                                           'AUC': cnn_results['auc']
                                                           }])], ignore_index=True)

        # 3. Naive Bayes
        nb_model = GaussianNB()
        nb_results = train_and_evaluate_with_cv(nb_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'NaiveBayes',
                                                           'Accuracy': nb_results['accuracy'],
                                                           'F1_Score': nb_results['f1_score'],
                                                           'Precision': nb_results['precision'],
                                                           'Recall': nb_results['recall'],
                                                           'AUC': nb_results['auc']
                                                           }])], ignore_index=True)

        # 4. Boosting (Gradient Boosting Classifier)
        gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb_results = train_and_evaluate_with_cv(gb_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'GradientBoosting',
                                                           'Accuracy': gb_results['accuracy'],
                                                           'F1_Score': gb_results['f1_score'],
                                                           'Precision': gb_results['precision'],
                                                           'Recall': gb_results['recall'],
                                                           'AUC': gb_results['auc']

                                                           }])], ignore_index=True)

        # 5. Random Forest
        rf_model = RandomForestClassifier(n_estimators=25, max_depth=7, min_samples_split=25, random_state=42)
        rf_results = train_and_evaluate_with_cv(rf_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'RandomForest',
                                                           'Accuracy': rf_results['accuracy'],
                                                           'F1_Score': rf_results['f1_score'],
                                                           'Precision': rf_results['precision'],
                                                           'Recall': rf_results['recall'],
                                                           'AUC': rf_results['auc']
                                                           }])], ignore_index=True)

        # 6. KNN (K-Nearest Neighbors)
        max_neighbors = min(2, X_train.shape[0] - 1)
        knn_model = KNeighborsClassifier(n_neighbors=max_neighbors)
        knn_results = train_and_evaluate_with_cv(knn_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'KNN',
                                                           'Accuracy': knn_results['accuracy'],
                                                           'F1_Score': knn_results['f1_score'],
                                                           'Precision': knn_results['precision'],
                                                           'Recall': knn_results['recall'],
                                                           'AUC': knn_results['auc']
                                                           }])], ignore_index=True)


        print(results_df)

        return results_df
    else:
        print("O tipo da comunidade não é 'Classification'. Nenhum modelo de classificação aplicado.")


def main_regression(class_community, sampling_strategy):
    if class_community['type'] == 'Regression':
        print("Aplicando modelos...")


        X = class_community['data'].iloc[:, :-1].reset_index(drop=True)
        y = class_community['data'].iloc[:, -1].reset_index(drop=True)

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

        transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ]
        )
        X_transformed = transformer.fit_transform(X)


        y_original = y.copy()

        scaler = RobustScaler()
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_scaled, test_size=0.3, random_state=42)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)


        if sampling_strategy is not None:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        def reg_train_and_evaluate(model, X_train, y_train, cv=5):
          kf = KFold(n_splits=cv, shuffle=True, random_state=42)

          spearman_scores = []

          for train_index, val_index in kf.split(X_train):

              X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
              y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

              model.fit(X_fold_train, y_fold_train)
              y_pred = model.predict(X_fold_val)

              y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
              y_fold_val_original = scaler.inverse_transform(y_fold_val.reshape(-1, 1)).flatten()

              spearman_corr, _ = spearmanr(y_fold_val_original, y_pred_original)
              spearman_scores.append(spearman_corr)


          return {
              'spearman': np.mean(spearman_scores),
          }


        results_df = pd.DataFrame(columns=['Model', 'Spearman'])

        # 1. Gradient Boosting Regressor
        gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=7, min_samples_split=5,
                                             min_samples_leaf=3, random_state=42)
        gb_results = reg_train_and_evaluate(gb_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'Gradient Boosting',
                                                           'Spearman': gb_results['spearman']
                                                           }])], ignore_index=True)

        # 2. Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=7, min_samples_split=15, random_state=42)
        rf_results = reg_train_and_evaluate(rf_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'Random Forest',
                                                           'Spearman': rf_results['spearman']
                                                           }])], ignore_index=True)

        # 3. K-Nearest Neighbors (KNN)
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_results = reg_train_and_evaluate(knn_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'K-Nearest Neighbors',
                                                           'Spearman': knn_results['spearman']
                                                           }])], ignore_index=True)

        #4 NegBin
        negbin_model = KerasRegressor(
            model=build_negbin_dnn,
            input_shape=X_train.shape[1],
            epochs=10,
            batch_size=32,
            verbose=0
        )
        negbin_results = reg_train_and_evaluate(negbin_model, X_train, y_train)
        results_df = pd.concat([results_df, pd.DataFrame([{'Model': 'NegBin',
                                                           'Spearman': negbin_results['spearman']
                                                           }])], ignore_index=True)

        print(results_df)
        return results_df

    else:
        print("O tipo da comunidade não é 'Regression'. Nenhum modelo de regressão aplicado.")


def main_real_data_importance(class_community, sampling_strategy):
    if class_community['type'] == 'Classification':
        print("Aplicando modelos...")

        X = class_community['data'].drop(columns=['X', 'Y', 'target'], errors='ignore')
        y = class_community['data'].iloc[:, -1].replace({'positive': 1.0, 'negative': 0.0})
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ]
        )
        X_transformed = transformer.fit_transform(X)
        feature_names = transformer.get_feature_names_out()
        
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
        
        if sampling_strategy is not None:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=2)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'AUC'])
        feature_importance = {}
        
 
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=25, max_depth=7, min_samples_split=25, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            feature_importance[model_name] = model.feature_importances_
        

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.mean(list(feature_importance.values()), axis=0)})
        

        importance_df['Original_Column'] = importance_df['Feature'].str.extract(r'cat__(.*?)_|num__(.*)')[0].fillna(
            importance_df['Feature'].str.extract(r'cat__(.*?)_|num__(.*)')[1]
        )
        
        consolidated_importance = importance_df.groupby('Original_Column').Importance.sum().reset_index()
        consolidated_importance = consolidated_importance.sort_values(by='Importance', ascending=False)
        
        print("\n### Importância das Features Consolidadas ###")
        print(consolidated_importance)
        
        return results_df, consolidated_importance
    else:
        print("O tipo da comunidade não é 'Classification'. Nenhum modelo de classificação aplicado.")



def run_multiple_classifications(class_community_list, sampling_c):
    for i, (class_community, sampling_strategy) in enumerate(zip(class_community_list, sampling_c)):
        if i > 3:
            print(f"Iniciando a classificação para: C4_{i-2}")
        else:
            print(f"Iniciando a classificação para: C{i+1}")
        results_df = main_classification(class_community, sampling_strategy)

        filename = f'C{i+1}.xlsx'
        results_df.to_excel(filename, index=False)
        print(f"Resultados salvos em: {filename}")
        

def run_real_data_classifications(class_community_list, sampling_c):
    for i, (class_community, sampling_strategy) in enumerate(zip(class_community_list, sampling_c)):
        print(f"Iniciando a classificação para Dados Reais")
        results_df = main_classification(class_community, sampling_strategy)
        filename = f'DadoReal.xlsx'
        results_df.to_excel(filename, index=False)
        print(f"Resultados salvos em: {filename}")


def run_multiple_regressions(class_community_list, sampling_r):
    for i, (class_community, sampling_strategy) in enumerate(zip(class_community_list, sampling_r)):

        if i > 3:
            print(f"Iniciando a regressão para: R4_{i-2}")
        else:
            print(f"Iniciando a regressão para: R{i+1}")

        results_df = main_regression(class_community, sampling_strategy)

        filename = f'R{i+1}.xlsx'
        results_df.to_excel(filename, index=False)
        print(f"Resultados salvos em: {filename}")