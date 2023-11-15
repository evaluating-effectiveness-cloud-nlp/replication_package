from utils import visualization
import pandas as pd
import sys
sys.path.append("../..")

metricsrq1 = pd.read_json(
    './outputs/experiment1/size99_07-12-2022 09_34_29/results/metrics.json')
metricsrq1 = metricsrq1[['noise_algorithm', 'provider',
                         'noise_level', 'fmeasure']]

metricsrq2 = pd.read_json(
    './outputs/experiment2/size100_07-04-2022 20_19_59/[5-10]/results/metrics.json')
metricsrq2 = metricsrq2[['noise_algorithm', 'provider',
                         'noise_level', 'fmeasure']]

metricsrq1 = metricsrq1.loc[metricsrq1['noise_algorithm'] != 'no_noise']
metricsrq2 = metricsrq2.loc[metricsrq2['noise_algorithm'] != 'no_noise']

noise_original_order = list(metricsrq1['noise_algorithm'].unique())
provider_original_order = list(metricsrq1['provider'].apply(
    lambda x: x.capitalize()).sort_values().unique())

output1: pd.DataFrame = visualization.save_summary_table(
    metricsrq1, "./test_scenarios/out/1_table", percent_noise=True)


output2: pd.DataFrame = visualization.save_latex_table(
    metricsrq2, "./test_scenarios/out/2_table", percent_noise=False)

output2
