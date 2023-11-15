from pathlib import Path
import pandas as pd
from . import latex, plots

def plot_results(results_array, main_path, noise_list, percent_noise=True):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure', 'confusion_matrix']]
    df = df[df['noise_algorithm'] != 'no_noise']

    plots.gen_results_plots_RQ1(results_array,
        main_path + '/rq1', noise_list)
    plots.gen_result_plot_RQ2_raw(results_array,
        main_path + '/rq2', noise_list)

    other_plots_path = main_path + '/others_plots'

    latex.gen_raw_values_table(df, other_plots_path, percent_noise)
    talex, table = latex.gen_summary_table(df, other_plots_path, percent_noise)

    df['noise_level']= df['noise_level'].map(str)

    save_raw_results_excel(df, main_path)
    save_raw_results_json(results_array, main_path)

    plots.save_confusion_matrix(df, other_plots_path)
    plots.save_results_plot(df, other_plots_path)

    return table.set_properties(**{'font-size': '16pt'})

def save_raw_results_json(results_array, main_path):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    filename = main_path + '/data.json'

    f = open(filename, "w")
    f.write(str(results_array))
    f.close()

def save_raw_results_excel(df, main_path):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    df['noise_level'] = df['noise_level'].map(float)
    df['fmeasure'] = df['fmeasure'].map(float)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure']]

    df = df.replace('_',' ', regex=True)

    df = df.rename(columns={
        "provider": "Provider",
        "noise_algorithm": "Noise Algorithm",
        "noise_level": "Noise Level",
        "fmeasure": "F-Measure"})

    filename = main_path + '/data_excel.xlsx'
    df.to_excel(filename, 'results', engine="openpyxl", index=False)
