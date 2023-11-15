from utils import visualization
import pandas as pd
import sys
sys.path.append("../..")

metricsrq12 = pd.read_json(
    './outputs/experiment1/size99_07-12-2022 09_34_29/results/metrics.json')

metricsrq3 = pd.read_json(
    './outputs/experiment2/size100_07-04-2022 20_19_59/[5-10]/results/metrics.json')

metricsrq12 = metricsrq12.to_dict('records')
metricsrq3 = metricsrq3.to_dict('records')


visualization.save_results_plot_RQ1(
    metricsrq12, "./test_scenarios/out/1", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).show()
visualization.save_results_plot_RQ1(
    metricsrq12, "./test_scenarios/out/1", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).show()
visualization.save_results_plot_RQ1(
    metricsrq3, "./test_scenarios/out/3", [1, 2, 3, 4, 5, 6, 7, 8, 9])

# visualization.save_results_plot_RQ2(metrics2, "./test_scenarios/out/2", [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9])
