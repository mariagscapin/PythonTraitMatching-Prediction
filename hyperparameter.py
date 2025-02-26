from machine_learning import *
from sklearn.model_selection import GridSearchCV

def build_dnn_hiper(input_shape, neurons=64, activation='relu', dropout_rate=0.2, optimizer='adam', l2_reg=0.01):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(neurons, activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons // 2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_cnn_hiper(input_shape, filters=64, kernel_size=2, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(input_shape, 1)))
    model.add(Conv1D(filters, kernel_size, activation=activation))
    model.add(Flatten())
    model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_negbin_dnn_hiper(input_shape, neurons=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons // 2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Saída para regressão
    model.compile(loss='poisson', optimizer=optimizer, metrics=['mse'])
    return model


def main_classification_hiper(class_community, sampling_strategy):
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

        if sampling_strategy is not None:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=2)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'AUC', 'Best_Params'])
        param_grids = {
            'RandomForest': {
                'n_estimators': [10, 25, 50, 100],
                'max_depth': [2, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 15]
            },
            'GradientBoosting': {
                'n_estimators': [25, 50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [2, 3, 5, 7, 10, None]
            },
            'KNN': {
                'n_neighbors': [2, 3, 5, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'NaiveBayes': {},
            'DNN': {
                  'model__neurons': [32, 64, 128],
                  'model__activation': ['relu', 'tanh'],
                  'model__dropout_rate': [0.2, 0.3],
                  'model__optimizer': ['adam', 'sgd'],
                  'epochs': [10, 15],
                  'batch_size': [16, 32]
              },

            'CNN': {
                  'model__filters': [32, 64],
                  'model__kernel_size': [2, 3],
                  'model__activation': ['relu', 'tanh'],
                  'model__optimizer': ['adam', 'sgd'],
                  'epochs': [10, 20],
                  'batch_size': [16, 32]
              }
        }


        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB(),
            'DNN': KerasClassifier(
                  model=build_dnn_hiper,
                  input_shape=X_train.shape[1],
                  verbose=0
              ),
            'CNN': KerasClassifier(
                    model=build_cnn_hiper,
                    input_shape=X_train.shape[1],
                    verbose=0)

         }

        for model_name, model in models.items():
            print(f"Treinando {model_name} com GridSearchCV...")
            if param_grids[model_name]:
                grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='f1', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = model.fit(X_train, y_train)
                best_params = None

            results = train_and_evaluate_with_cv(best_model, X_train, y_train)
            results['Best_Params'] = best_params

            results_df = pd.concat([results_df, pd.DataFrame([{
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1_Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'AUC': results['auc'],
                'Best_Params': results['Best_Params']
            }])], ignore_index=True)

        print(results_df)


        return results_df
    else:
        print("O tipo da comunidade não é 'Classification'. Nenhum modelo de classificação aplicado.")


def main_regression_hiper(class_community, sampling_strategy):
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

        scaler = RobustScaler()
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_scaled, test_size=0.3, random_state=42)

        if sampling_strategy is not None:
            smote = SMOTE(sampling_strategy= sampling_strategy, random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        def reg_train_and_evaluate(model, X_train, y_train, cv=3):
          kf = KFold(n_splits=cv, shuffle=True, random_state=42)

          spearman_scores = []
          r2_scores = []
          tss_scores = []

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


        param_grids = {
            'RandomForest': {
                'n_estimators': [25, 50, 100],
                'max_depth': [2, 5, 10],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 3, 5]
            },
            'KNN': {
                'n_neighbors': [2, 3, 5, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'NegBin': {
                'neurons': [32, 64, 128],
                'activation': ['relu', 'tanh'],
                'optimizer': ['adam', 'sgd'],
                'epochs': [10, 20],
                'batch_size': [16, 32]
            },
            'GradientBoosting': {
                'n_estimators': [25, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5]
            },
        }


        results_df = pd.DataFrame(columns=['Model', 'Spearman', 'Best_Params'])

        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'KNN': KNeighborsRegressor(),
            'NegBin': KerasRegressor(model=build_negbin_dnn_hiper,
                                     activation='relu',
                                     optimizer='adam',
                                     epochs=10,
                                     batch_size=32,
                                     neurons=64,
                                     input_shape=X_train.shape[1], verbose=0),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
        }

        for model_name, model in models.items():
            print(f"Treinando {model_name} com GridSearchCV...")
            if param_grids[model_name]:
                grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = model.fit(X_train, y_train)
                best_params = None


            results = reg_train_and_evaluate(best_model, X_train, y_train)
            results['Best_Params'] = best_params

            results_df = pd.concat([results_df, pd.DataFrame([{
                'Model': model_name,
                'Spearman': results['spearman'],
                'Best_Params': results['Best_Params']
            }])], ignore_index=True)

        return results_df
    else:
        print("O tipo da comunidade não é 'Regression'. Nenhum modelo de regressão aplicado.")

def run_multiple_regressions_hiper(class_community_list, sampling_r):
    for i, (class_community, sampling_strategy) in enumerate(zip(class_community_list, sampling_r)):

        if i > 3:
            print(f"Iniciando a regressão para: R4_{i-2}")
        else:
            print(f"Iniciando a regressão para: R{i+4}")

        results_df = main_regression_hiper(class_community, sampling_strategy)

        filename = f'R{i+1}.xlsx'
        results_df.to_excel(filename, index=False)
        print(f"Resultados salvos em: {filename}")

def run_multiple_classification_hiper(class_community_list, sampling_r):
    for i, (class_community, sampling_strategy) in enumerate(zip(class_community_list, sampling_r)):

        if i > 3:
            print(f"Iniciando a regressão para: C4_{i-2}")
        else:
            print(f"Iniciando a regressão para: C{i+4}")

        results_df = main_classification_hiper(class_community, sampling_strategy)

        filename = f'C{i+1}.xlsx'
        results_df.to_excel(filename, index=False)
        print(f"Resultados salvos em: {filename}")

