from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from pathlib import Path
import json
import pandas as pd

def save_metrics_to_file(main_path, metrics):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    with open(main_path+'/metrics.json', 'w') as f:
        json.dump(metrics, f)

def map_input(value):
    if value == 'negative':
        return -1
    if value == 'positive':
        return 1
    if value == 'neutral':
        return 0

    return None

def calc_dataset_metrics(y_labels, predicted_labels):
    # Transformando os labels em númericos para analise de metricas:
    y_labels_binary = list(map(map_input, y_labels))
    predicted_binary = list(map(map_input, predicted_labels))
    acc = accuracy_score(y_labels_binary,predicted_binary)
    recall = recall_score(y_labels_binary,predicted_binary, average="weighted")
    precision = precision_score(y_labels_binary,predicted_binary, average="weighted", zero_division=0)
    auc = None #roc_auc_score(y_labels_binary,(predicted_binary, 3), multi_class="ovr", average="weighted") precisa de ajustes para multi class
    fmeasure = f1_score(y_labels_binary,predicted_binary, average="weighted", zero_division=0)
    confusion_m = confusion_matrix(y_labels_binary, predicted_binary)
    # confusion_m_mult = multilabel_confusion_matrix(y_labels_binary, predicted_binary) #, labels=["ne", "bird", "cat"]

    return (acc, recall, precision, auc, fmeasure, confusion_m)

def print_metrics(metrics_dict):
    acc = metrics_dict['acc']
    precision  = metrics_dict['precision']
    recall = metrics_dict['recall']
    auc = metrics_dict['auc']
    
    print(f'Azure MLaaS Metrics', sep="\n")
    print(f'Accuracy = {acc} ## Precision = {precision} ## Recall = {recall} ## AUC = {auc}')
    print('----------------------------------------------------------------------------------')

def load_predictions(path):
    dataframe = pd.read_excel(path, engine='openpyxl')
    predictions = dataframe.values.tolist()
    predictions = [p[0] for p in predictions]

    return predictions

def metrics(progress, y_labels, main_path):
    metrics = []

    providers = list(progress['predictions'].keys())

    for provider in providers:
        noises = list(progress['predictions'][provider].keys())
        for noise in noises:
            noise_levels_list = list(progress['predictions'][provider][noise].keys())
            noise_levels_list = [float(l) for l in noise_levels_list]

            for level in noise_levels_list:
                predictions_path = progress['predictions'][provider][noise][str(level)]
                predictions = load_predictions(predictions_path)

                if len(y_labels) != len(predictions):
                    print('inconsistent values at: ', provider, noise, level)
                acc, recall, precision, auc, fmeasure, confusion_m = calc_dataset_metrics(y_labels,predictions)
                result = {'provider':provider,
                        'noise_algorithm':noise,
                        'noise_level':0 if noise == 'no_noise' else level,
                        'acc':acc, 'recall':recall, 'precision': precision, 'auc': auc, 'fmeasure': fmeasure,
                        'confusion_matrix': confusion_m.tolist()
                }
                metrics.append(result)
    save_metrics_to_file(main_path+'/results/', metrics)
    return metrics
