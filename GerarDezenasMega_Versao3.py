import logging
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.utils import resample, estimator_html_repr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier # Redes neurais
from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
import gc
import psutil
import sys

# Exibir o uso de memória
print(f"Uso de memória: {psutil.virtual_memory().percent}%")

# Configuração de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('program_log.log', mode='w')]
)

# Função para logar tempo de execução
def log_tempo_execucao(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Marca o tempo de início da execução
        result = func(*args, **kwargs)  # Chama a função original
        end_time = time.time()  # Marca o tempo de término da execução
        execution_time = end_time - start_time  # Calcula o tempo de execução
        logging.info(f"{func.__name__} executada em {execution_time:.4f} segundos.")  # Registra o tempo de execução
        return result  # Retorna o resultado da função original
    return wrapper

# Função para carregar arquivos e retornar um DataFrame
@log_tempo_execucao
def carregar_dataframe(caminho_arquivo, tipo):
    """
    Carrega um DataFrame de um arquivo pickle (.pkl).
    
    Parâmetros:
    caminho_arquivo (str): Caminho do arquivo .pkl.
    
    Retorna:
    pd.DataFrame: DataFrame carregado do arquivo.
    
    Lança:
    FileNotFoundError: Se o arquivo não for encontrado.
    ValueError: Se o arquivo estiver vazio.
    """
    try:
        if not os.path.exists(caminho_arquivo):
            logging.error(f"Arquivo não encontrado: {caminho_arquivo}")
            return None

        if not caminho_arquivo.endswith('.pkl'):
            logging.error(f"Arquivo inválido. Esperado um PKL, mas recebeu: {caminho_arquivo}")
            return None

        logging.info(f"Iniciando leitura do arquivo .pkl: {caminho_arquivo}")
        
        if tipo == 'pkl':
            df = pd.read_pickle(caminho_arquivo)
            logging.info(f"Dataframe carregado com sucesso. Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
            if df is None:
                raise ValueError(f"O arquivo {caminho_arquivo} está vazio ou corrompido.")
            
        elif tipo == 'csv':
            df = pd.read_csv(caminho_arquivo)
            logging.info(f"Dataframe carregado com sucesso. Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
            if df is None:
                raise ValueError(f"O arquivo {caminho_arquivo} está vazio ou corrompido.")

        if df.empty:
            raise ValueError(f"O arquivo {caminho_arquivo} está vazio.")
        
        logging.info(f"Arquivo {caminho_arquivo} carregado com sucesso. Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
        return df

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as e:
        logging.error(f"Erro ao carregar o arquivo {caminho_arquivo}: {e}")
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo {caminho_arquivo}: {e}")
    return None

# Função para calcular frequências e salvar em arquivo
@log_tempo_execucao
def calcular_frequencias_e_salvar(df_resultados, arquivo_saida="frequencias.txt"):
    """
    Calcula as frequências de cada número sorteado, exibe o resultado e salva em um arquivo .txt.
    
    Parâmetros:
    df_resultados (pd.DataFrame): DataFrame com os resultados dos sorteios.
    arquivo_saida (str): Caminho para o arquivo onde as frequências serão salvas.
    
    Retorna:
    dict: Dicionário com os números sorteados como chave e a frequência como valor.
    """
    try:
        # Calcular as frequências
        frequencias = df_resultados.apply(pd.Series.value_counts, axis=1).sum(axis=0)
        frequencias_dict = frequencias.to_dict()

        # Exibir as frequências no console
        print("Frequências calculadas:")
        for numero, frequencia in frequencias_dict.items():
            print(f"Número: {numero}, Frequência: {frequencia}")

        # Salvar as frequências em um arquivo .txt
        with open(arquivo_saida, "w") as arquivo:
            arquivo.write("Número, Frequência\n")
            for numero, frequencia in frequencias_dict.items():
                arquivo.write(f"{numero}, {frequencia}\n")
        
        logging.info("Frequências salvas com sucesso.")
        return frequencias_dict

    except Exception as e:
        logging.error(f"Erro ao calcular ou salvar frequências: {e}")
        raise

# Função para balanceamento de classes
@log_tempo_execucao
def balancear_classes(X, y):
    """
    Balanceia as classes utilizando SMOTE e downsampling.
    """
    try:
        # Usar SMOTE para oversampling
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_smote, y_smote = smote.fit_resample(X, y)
        logging.info(f"Classes balanceadas com SMOTE: {Counter(y_smote)}")
        
        # Fazer downsampling da classe majoritária
        X_balanced, y_balanced = downsample_classes(X_smote, y_smote)
        logging.info(f"Classes após downsampling: {Counter(y_balanced)}")

        return X_balanced, y_balanced

    except Exception as e:
        logging.error(f"Erro ao balancear classes: {e}")
        raise

# Função para treinar o modelo
def treinar_modelo(X_train, y_train, modelo):
    """
    Treina o modelo com os dados fornecidos.
    """
    modelo.fit(X_train, y_train)
    return modelo

# Classe para gerenciamento de modelos
class Modelo:
    def __init__(self, modelo):
        self.modelo = modelo

    @log_tempo_execucao
    def treinar(self, X_train, y_train):
        """
        Treina o modelo com os dados fornecidos.
        """
        try:
            with tqdm(total=1, desc=f"Treinando modelo {self.modelo.__class__.__name__}") as pbar:
                self.modelo.fit(X_train, y_train)
                pbar.update(1)
            logging.info(f"Modelo {self.modelo.__class__.__name__} treinado com sucesso.")
            return self.modelo
        except Exception as e:
            logging.error(f"Erro ao treinar o modelo {self.modelo.__class__.__name__}: {e}")
            raise

    @log_tempo_execucao
    def avaliar(self, X_test, y_test):
        """
        Avalia a performance do modelo nos dados de teste.
        """
        try:
            score = self.modelo.score(X_test, y_test)
            logging.info(f"Modelo {self.modelo.__class__.__name__} obteve um score de {score:.4f}.")
            return score
        except Exception as e:
            logging.error(f"Erro ao avaliar o modelo {self.modelo.__class__.__name__}: {e}")
            raise

    @log_tempo_execucao
    def ajustar_hiperparametros(self, X_train, y_train, param_grid):
        """
        Ajusta os hiperparâmetros do modelo usando GridSearchCV.
        """
        try:
            grid_search = GridSearchCV(self.modelo, param_grid, cv=5)
            with tqdm(total=len(param_grid) * 5, desc="Ajustando hiperparâmetros") as pbar:
                grid_search.fit(X_train, y_train)
                pbar.update()
            logging.info(f"Melhor parâmetro encontrado: {grid_search.best_params_}")
            return grid_search.best_estimator_

        except Exception as e:
            logging.error(f"Erro ao ajustar hiperparâmetros para {self.modelo.__class__.__name__}: {e}")
            raise

# Função para preparar os dados de entrada para o modelo
@log_tempo_execucao
def preparar_dados(frequencias, combinacoes_existentes):
    """
    Prepara os dados de entrada para o modelo, incluindo as combinações existentes e aleatórias.
    
    Parâmetros:
    frequencias (dict): Frequências de cada número sorteado.
    combinacoes_existentes (set): Conjunto de combinações já existentes.
    
    Retorna:
    np.ndarray: Matrizes de dados X (características) e y (rótulos).
    
    Lança:
    Exception: Se ocorrer algum erro ao preparar os dados.
    """
    X, y = [], []
    try:
        # Adiciona combinações existentes
        for comb in tqdm(combinacoes_existentes, desc="Preparando combinações existentes", unit="combinação"):
            frequencias_comb = [frequencias.get(num, 0) for num in comb]
            X.append(frequencias_comb)
            y.append(1)  # 1 para combinações existentes

        # Adiciona combinações aleatórias
        n_combinacoes_aleatorias = 100  # Gerar 100 combinações aleatórias
        combinacoes_aleatorias = set()  # Usar set para evitar duplicação
        while len(combinacoes_aleatorias) < n_combinacoes_aleatorias:
            comb_aleatoria = tuple(sorted(np.random.choice(list(frequencias.keys()), size=6, replace=False)))
            if comb_aleatoria not in combinacoes_existentes:
                combinacoes_aleatorias.add(comb_aleatoria)

        for comb_aleatoria in tqdm(combinacoes_aleatorias, desc="Gerando combinações aleatórias", unit="combinação"):
            frequencias_comb_aleatoria = [frequencias.get(num, 0) for num in comb_aleatoria]
            X.append(frequencias_comb_aleatoria)
            y.append(0)  # 0 para combinações aleatórias

        logging.info(f"Preparação de dados concluída: {len(X)} combinações no total.")
        return np.array(X), np.array(y)
    except Exception as e:
        logging.error(f"Erro ao preparar dados: {e}")
        raise

# Função para balanceamento das classes (downsampling)
@log_tempo_execucao
def downsample_classes(X, y):
    """
    Balanceia as classes de combinações existentes e aleatórias, realizando downsampling.
    
    Parâmetros:
    X (np.ndarray): Características (entrada).
    y (np.ndarray): Rótulos (saída).
    
    Retorna:
    np.ndarray: Dados balanceados X e y.
    
    Lança:
    Exception: Se ocorrer algum erro ao balancear as classes.
    """
    try:
        logging.info("Balanceando as classes (downsampling)...")
        
        # Divida as classes
        X_existentes = X[y == 1]
        X_aleatorias = X[y == 0]
        
        # Balanceando as classes
        X_existentes_balanced, y_existentes_balanced = resample(X_existentes, y[y == 1], replace=False, 
                                                                n_samples=len(X_aleatorias), random_state=42)
        
        X_balanced = np.concatenate([X_existentes_balanced, X_aleatorias], axis=0)
        y_balanced = np.concatenate([y_existentes_balanced, y[y == 0]], axis=0)
        
        logging.info(f"Classes balanceadas: {len(X_balanced)} amostras no total.")
        return X_balanced, y_balanced
    except Exception as e:
        logging.error(f"Erro ao balancear classes: {e}")
        raise

# Após o downsampling ou operações de balanceamento, faça:
gc.collect()  # Limpa a memória de objetos não utilizados

@log_tempo_execucao
def gerar_e_salvar_combinacoes(frequencias, caminho_csv_salvar, caminho_pkl_salvar, combinacoes_existentes):
    """
    Gera novas combinações e as salva em arquivos CSV e pickle.
    
    Parâmetros:
    frequencias (dict): Frequências dos números sorteados.
    caminho_csv_salvar (str): Caminho para salvar o arquivo CSV.
    caminho_pkl_salvar (str): Caminho para salvar o arquivo pickle.
    combinacoes_existentes (set): Conjunto de combinações já existentes.
    
    Lança:
    Exception: Se ocorrer algum erro ao gerar ou salvar as combinações.
    """
    try:
        if os.path.exists(caminho_pkl_salvar):
            logging.info("Carregando combinações anteriores do arquivo .pkl...")
            df_combinacoes_existentes = pd.read_pickle(caminho_pkl_salvar)
            combinacoes_existentes = set(tuple(sorted(comb)) for comb in df_combinacoes_existentes.values)
            logging.info(f"{len(combinacoes_existentes)} combinações anteriores carregadas com sucesso.")
        else:
            logging.info("Nenhuma combinação anterior encontrada, gerando novas combinações.")
            combinacoes_existentes = set()

        combinacoes = []
        for _ in tqdm(range(100), desc="Gerando combinações para salvar", unit="combinação"):
            nova_combinacao = tuple(sorted(np.random.choice(list(frequencias.keys()), size=6, replace=False)))
            if nova_combinacao not in combinacoes_existentes:
                combinacoes.append(nova_combinacao)

        pd.DataFrame(combinacoes).to_csv(caminho_csv_salvar, index=False, header=False)
        logging.info(f"{len(combinacoes)} combinações geradas e salvas em {caminho_csv_salvar}.")

        df_combinacoes = pd.DataFrame(combinacoes)
        df_combinacoes.to_pickle(caminho_pkl_salvar)
        logging.info(f"Combinações também salvas em {caminho_pkl_salvar}.")
    except Exception as e:
        logging.error(f"Erro ao gerar ou salvar combinações: {e}")
        raise

@log_tempo_execucao
def main():
    """
    Função principal que executa o fluxo do programa:
    - Carregar dados.
    - Calcular frequências.
    - Preparar dados.
    - Balancear classes.
    - Treinar e avaliar modelos de aprendizado de máquina.
    - Gerar e salvar novas combinações.
    """
    try:
        caminho_df_resultados = 'Mega-Sena-Sorteios.pkl'
        caminho_df_combinacoes = 'df_combinacoes.pkl'
        caminho_csv_salvar = './combinacoes_geradas.csv'
        caminho_pkl_salvar = './combinacoes_geradas.pkl'  # Novo arquivo .pkl

        df_resultados = carregar_dataframe(caminho_df_resultados, tipo='pkl')
        combinacoes_existentes = carregar_dataframe(caminho_df_combinacoes, tipo='pkl')

        frequencias = calcular_frequencias_e_salvar(df_resultados)
        combinacoes_existentes = set(tuple(sorted(row)) for row in combinacoes_existentes.values)
        X, y = preparar_dados(frequencias, combinacoes_existentes)
        
        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Balancear classes
        X_train_balanced, y_train_balanced = balancear_classes(X_train, y_train)

        # Inicializar modelos
        modelo_naive_bayes = Modelo(GaussianNB())
        modelo_knn = Modelo(KNeighborsClassifier())
        modelo_decision_tree = Modelo(DecisionTreeClassifier())
        modelo_svc = Modelo(SVC())
        modelo_random_forest = Modelo(RandomForestClassifier())
        modelo_mlp = Modelo(MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000))  # Redes neurais

        # Treinamento dos modelos
        modelo_naive_bayes.treinar(X_train_balanced, y_train_balanced)
        modelo_knn.treinar(X_train_balanced, y_train_balanced)
        modelo_decision_tree.treinar(X_train_balanced, y_train_balanced)
        modelo_svc.treinar(X_train, y_train)
        modelo_random_forest.treinar(X_train_balanced, y_train_balanced)
        modelo_mlp.treinar(X_train_balanced, y_train_balanced)

        # Avaliar os modelos
        modelo_naive_bayes.avaliar(X_test, y_test)
        #modelo_knn.avaliar(X_test, y_test)
        #modelo_decision_tree.avaliar(X_test, y_test)
        #modelo_svc.avaliar(X_test, y_test)
        #modelo_random_forest.avaliar(X_test, y_test)
        #modelo_mlp.avaliar(X_test, y_test)
        
        # Modelos de aprendizado de máquina
        modelos = {
            "Naive Bayes": GaussianNB(),
            "Árvore de Decisão": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            #"KNN": KNeighborsClassifier(),
            #"MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)  # Redes neurais
        }

        for nome_modelo, modelo in modelos.items():
            logging.info(f"Iniciando validação cruzada para {nome_modelo}")
            scores = cross_val_score(modelo, X, y, cv=5)
            logging.info(f"Acurácia média de {nome_modelo} (validação cruzada): {scores.mean():.4f}")

            modelo_treinado = treinar_modelo(X_train, y_train, modelo)
            y_pred = modelo_treinado.predict(X_test)
            
            acuracia = accuracy_score(y_test, y_pred)
            precisao = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            matriz_confusao = confusion_matrix(y_test, y_pred)
            
            logging.info(f"Acurácia de {nome_modelo} no conjunto de teste: {acuracia:.4f}")
            logging.info(f"Precisão de {nome_modelo}: {precisao:.4f}")
            logging.info(f"Recall de {nome_modelo}: {recall:.4f}")
            logging.info(f"F1-Score de {nome_modelo}: {f1:.4f}")
            logging.info(f"Matriz de Confusão de {nome_modelo}:\n{matriz_confusao}")
            
        gerar_e_salvar_combinacoes(frequencias, caminho_csv_salvar, caminho_pkl_salvar, combinacoes_existentes)
        logging.info("Processo concluído com sucesso.")

    except Exception as e:
        logging.error(f"Erro no fluxo principal: {e}")
        raise

if __name__ == "__main__":
    main()

    # Após concluir o trabalho do programa
    gc.collect()  # Força a coleta de lixo para liberar memória não utilizada

    # Para fechar o programa corretamente
    sys.exit("Programa encerrado e memória liberada.")
