# create_species.py

Este script contém a função principal `create_species` e auxiliares para gerar espécies e suas características com base em distribuições probabilísticas. O código permite a criação de dois grupos de espécies (`A` e `B`) com diferentes características genéticas e ambientais, como abundância e especialização.

## Funcionalidade

### Classe `SpeciesClass`
A classe `SpeciesClass` armazena as informações geradas sobre duas espécies (`A` e `B`), incluindo características genéticas, abundância e outros atributos relacionados à especialização.

### Função `create_discrete`
A função `create_discrete` gera uma função de amostragem discreta com base em um intervalo fornecido, retornando amostras baseadas em probabilidades normalizadas.

### Função `create_species`
A função principal `create_species` gera duas espécies (`A` e `B`) com características especificadas. Ela pode ser configurada para incluir especialização (espécies mais adaptadas a certos ambientes) e personalização nas distribuições de abundância e características.

## Parâmetros da função `create_species`
- `NumberA`: Número de indivíduos da espécie A.
- `NumberB`: Número de indivíduos da espécie B.
- `traitsA`: Lista com dois valores indicando os tipos de características para a espécie A.
- `traitsB`: Lista com dois valores indicando os tipos de características para a espécie B.
- `abundance`: Função ou valor que define a abundância das espécies.
- `specRange`: Intervalo de especialização para a espécie B.
- `speciesClass`: Instância de `SpeciesClass` para herdamento de características (opcional).
- `specialist`: Define se a espécie é especialista ou generalista.
- `coLin`: Parâmetro de variáveis colineares.

# simulate_interaction.py

Este script simula interações ecológicas entre duas espécies, levando em consideração suas características e traços, como a abundância e a especialização. A interação entre as espécies é modelada com base em funções de distribuição (Poisson ou binária) e cálculos de interações específicas entre os indivíduos das duas espécies.

## Funções Principais

- **`check_weights(weights, main_len, inter_len)`**: Verifica se o dicionário de pesos contém chaves válidas e se os comprimentos das listas de pesos correspondem aos tamanhos esperados.
  
- **`check_species(species, kwargs)`**: Verifica a existência da instância de uma espécie e cria uma nova, se necessário.
  
- **`calculate_trait_value(val, x, discrete, mainTraits, spec=0.5)`**: Calcula o valor de um traço para um indivíduo, considerando se o traço é discreto ou contínuo.
  
- **`create_distribution_func(interMatrix, is_poisson=True, x=1000, seed=42)`**: Cria uma função para gerar distribuições de Poisson ou binárias com base na matriz de interações.

- **`calculate_interactions_logic(i, x, y, inter, interTraits, spec=0.5)`**: Calcula o valor de uma interação entre indivíduos com base em suas características e os traços definidos.

- **`calculate_interactions(species, mainFunc, interFunc, inter, interTraits, weights, interMatrix)`**: Calcula a matriz de interações entre as duas espécies, otimizando os cálculos de ponto flutuante.

- **`simulate_interaction(species, main, inter, weights={ "main": [], "inter": []}, re_sim=None, **kwargs)`**: Função principal que simula as interações entre duas espécies, gerando a matriz de interações, funções de distribuição, e outros resultados relacionados às interações ecológicas.

## Parâmetros da função `simulate_interaction`

- `species`: Instância de `SpeciesClass` com as informações sobre as espécies A e B.
- `main`: Lista de traços principais para as espécies.
- `inter`: Lista de interações entre as espécies.
- `weights`: Dicionário contendo os pesos para os traços principais e interações.
- `re_sim`: Dados para re-simulação de interações (opcional).
- `kwargs`: Outros parâmetros adicionais.

# create_community.py

Este script cria uma comunidade ecológica simulada a partir de dados de interações entre espécies. Ele permite a imputação de valores ausentes e gera uma estrutura de dados para modelagem, levando em consideração as interações entre as espécies, traços, e respostas. O código também lida com a criação de uma matriz de interações e classifica a tarefa em "Classificação" ou "Regressão", dependendo da natureza do alvo.

## Funções Principais

- **`createCommunity(a=None, b=None, z=None, community=None, response=None, positive=None, impute=True, log=True)`**: Função principal que cria uma comunidade ecológica, processando dados de interações entre espécies e imputando dados ausentes. Retorna um dicionário com informações sobre os dados e o tipo de tarefa (classificação ou regressão).

- **`create_inter(imp_data, z, log)`**: Cria a matriz de interações entre as espécies com base nos dados imputados, combinando as informações das espécies A e B.

- **`impute_data(a, b)`**: Imputa dados ausentes nas interações de duas espécies (A e B) usando o algoritmo IterativeImputer do `sklearn`.

- **`check_input(a=None, b=None, z=None, community=None, response=None, positive=None)`**: Verifica a validade das entradas fornecidas para a criação da comunidade, garantindo que os dados necessários estejam presentes.

## Parâmetros da função `createCommunity`

- `a`, `b`: DataFrames com as interações das duas espécies (A e B).
- `z`: Matriz de interações entre as espécies.
- `community`: Lista ou DataFrame contendo informações sobre a comunidade a ser criada.
- `response`: Nome da coluna alvo para a tarefa de modelagem (classificação ou regressão).
- `positive`: Valor da classe positiva para classificação binária.
- `impute`: Define se os dados ausentes devem ser imputados (padrão: `True`).
- `log`: Define se os dados devem ser transformados em log (padrão: `True`).

# machine_learning.py

Este script contém funções e modelos para realizar tarefas de classificação e regressão utilizando diferentes algoritmos de aprendizagem de máquina. O código implementa pipelines de pré-processamento, treinamento, avaliação e importação de dados reais.

## Bibliotecas Utilizadas
- `sklearn`: para modelos de Machine Learning, pré-processamento de dados e métricas.
- `tensorflow.keras`: para a construção e treinamento de redes neurais.
- `imblearn`: para lidar com desbalanceamento de classes utilizando SMOTE.
- `scipy`: para cálculo de correlação de Spearman.
- `pandas` e `numpy`: para manipulação e processamento de dados.

## Funções Principais

### Funções de Construção de Modelos

- **`build_dnn(input_shape)`**: Cria um modelo de rede neural densa (DNN) para classificação binária com regularização L2 e dropout.
- **`build_cnn(input_shape)`**: Cria um modelo de rede neural convolucional (CNN) para classificação binária.
- **`build_negbin_dnn(input_shape)`**: Cria um modelo de rede neural para regressão utilizando a distribuição Poisson.

### Funções de Treinamento e Avaliação

- **`train_and_evaluate_with_cv(model, X_train, y_train, cv=5)`**: Realiza validação cruzada estratificada para avaliar o modelo de classificação utilizando métricas como Acurácia, F1 Score, Precisão, Recall e AUC.
  
### Funções de Classificação e Regressão

#### **Classificação**

- **`main_classification(class_community, sampling_strategy)`**: Aplica modelos de classificação (DNN, CNN, Naive Bayes, Random Forest, Gradient Boosting, KNN) em dados de uma comunidade e avalia os resultados com validação cruzada.
  
#### **Regressão**

- **`main_regression(class_community, sampling_strategy)`**: Aplica modelos de regressão (Gradient Boosting, Random Forest, KNN) e avalia a correlação de Spearman entre valores reais e previstos.
  
#### **Importância das Features em Dados Reais**

- **`main_real_data_importance(class_community, sampling_strategy)`**: Aplica modelos de classificação e calcula a importância das features utilizando Random Forest e Gradient Boosting em dados reais.

### Funções para Executar Múltiplas Simulações

- **`run_multiple_classifications(class_community_list, sampling_c)`**: Executa classificações para múltiplas comunidades de dados e salva os resultados em arquivos `.xlsx`.
- **`run_real_data_classifications(class_community_list, sampling_c)`**: Executa classificações em dados reais e salva os resultados em arquivos `.xlsx`.
- **`run_multiple_regressions(class_community_list, sampling_r)`**: Executa regressões para múltiplas comunidades de dados e salva os resultados em arquivos `.xlsx`.

# hyperparameter.py

Este script contém funções para realizar ajustes de hiperparâmetros de modelos de classificação e regressão utilizando `GridSearchCV`. O objetivo é otimizar os parâmetros de diversos algoritmos de Machine Learning, incluindo redes neurais e modelos clássicos, para encontrar a melhor combinação que maximize o desempenho.

## Bibliotecas Utilizadas
- `sklearn`: para otimização de hiperparâmetros e modelos de Machine Learning.
- `tensorflow.keras`: para construção e treinamento de redes neurais.
- `imblearn`: para lidar com desbalanceamento de classes utilizando SMOTE.
- `pandas` e `numpy`: para manipulação de dados.

## Funções Principais

### Funções de Construção de Modelos com Hiperparâmetros

- **`build_dnn_hiper(input_shape, neurons=64, activation='relu', dropout_rate=0.2, optimizer='adam', l2_reg=0.01)`**: Cria um modelo de rede neural densa (DNN) para classificação binária, com regularização L2 e dropout.
- **`build_cnn_hiper(input_shape, filters=64, kernel_size=2, activation='relu', optimizer='adam')`**: Cria um modelo de rede neural convolucional (CNN) para classificação binária.
- **`build_negbin_dnn_hiper(input_shape, neurons=64, activation='relu', optimizer='adam')`**: Cria um modelo de rede neural para regressão utilizando a distribuição Poisson.

### Funções de Treinamento e Avaliação com Ajuste de Hiperparâmetros

- **`main_classification_hiper(class_community, sampling_strategy)`**: Aplica modelos de classificação (Random Forest, Gradient Boosting, KNN, Naive Bayes, DNN e CNN) com ajuste de hiperparâmetros utilizando `GridSearchCV` e avalia o desempenho em termos de Acurácia, F1 Score, Precisão, Recall e AUC.
  
- **`main_regression_hiper(class_community, sampling_strategy)`**: Aplica modelos de regressão (Random Forest, KNN, NegBin e Gradient Boosting) com ajuste de hiperparâmetros utilizando `GridSearchCV` e avalia a correlação de Spearman entre valores reais e previstos.

### Funções para Executar Múltiplas Simulações

- **`run_multiple_regressions_hiper(class_community_list, sampling_r)`**: Executa múltiplas regressões em uma lista de comunidades de dados e salva os resultados em arquivos `.xlsx`.
- **`run_multiple_classification_hiper(class_community_list, sampling_r)`**: Executa múltiplas classificações em uma lista de comunidades de dados e salva os resultados em arquivos `.xlsx`.




