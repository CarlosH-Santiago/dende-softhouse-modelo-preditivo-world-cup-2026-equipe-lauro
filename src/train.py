
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def treinar_modelo_copa(X, y):
  
    print("Iniciando a Task 2: Treinamento e Validação do Modelo...")
    
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dados divididos: conjunto de treino e conjunto de validação criados.")

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Algoritmo Random Forest instanciado.")
    
    modelo.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")

    y_pred = modelo.predict(X_val)

    acuracia = accuracy_score(y_val, y_pred)
    matriz_conf = confusion_matrix(y_val, y_pred)
    
    print("\n" + "="*40)
    print("MÉTRICAS DE AVALIAÇÃO DO MODELO")
    print("="*40)
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print("\nMatriz de Confusão:")
    print(matriz_conf)
    print("\nRelatório de Classificação Completo:")
    print(classification_report(y_val, y_pred))
    
    return modelo

if __name__ == "__main__":

    
    print("Para executar este arquivo diretamente, é necessário importar X e y do data_prep.py")