import pandas as pd
from pathlib import Path
from utils.dataframe import divide_dataframe

pd.set_option('mode.chained_assignment', None)

def _format_float(number: float):
    f = '{0:.2g}'.format(number)
    return f

def _prepare_table_to_latex(df, percent=True):
    df = df[['provider', 'noise_algorithm', 'noise_level', 'fmeasure']]
    df.loc[:, 'noise_level'] = df['noise_level'].map(float)

    if percent:
        df.loc[:, 'noise_level'] = df['noise_level'].map(lambda a: a * 100)
        df.loc[:, 'noise_level'] = df['noise_level'].map(_format_float)
    else:
        df.loc[:, 'noise_level'] = df['noise_level'].map(int)

    df = df.replace('_', ' ', regex=True)
    df.provider = df.provider.str.capitalize()

    df = df.rename(columns={
        "provider": "Provider",
        "noise_algorithm": "Noise Algorithm",
        "noise_level": "Noise Level",
        "fmeasure": "F-Measure"})
    
    return df

def gen_summary_table(df: pd.DataFrame, main_path, percent_noise):             
    def sorting(item, noise_order, provider_order):
        index = None
        try:
            index = noise_order.index(item)
        except:
            index = provider_order.index(item)
        return index

    Path(main_path).mkdir(parents=True, exist_ok=True)
    
    df = _prepare_table_to_latex(df, percent_noise)

    noise_order =list(df["Noise Algorithm"].unique())
    provider_order = list(df['Provider'].apply(lambda x: x.capitalize())
                          .sort_values()
                          .unique())
    noise_column_name = 'Noise Level' + (' (%)' if percent_noise else ' (UN)')
    df = df.rename(columns={"Noise Level": noise_column_name})

    df = df[['Noise Algorithm', 'Provider',
             noise_column_name, 'F-Measure']]

    df = df.pivot(values='F-Measure', index=[
        'Noise Algorithm', 'Provider'], columns=noise_column_name)
    df = df.sort_values(by=['Noise Algorithm', 'Provider'],
                        key=lambda x: x.map(lambda s: sorting(s, noise_order, provider_order)))

    df.columns = pd.MultiIndex.from_tuples(
        [(noise_column_name, noise) for noise in df.columns])
    
    df.index.names = [None, None]
    filename = main_path + f'/table_latex_rq2_summary.txt'

    table = df.style \
        .background_gradient(cmap='RdYlGn', axis=None, low=0, high=1.0) \
        .set_properties(**{'font-size': '4px'}) \
        .format(precision=2) \
        .applymap_index(lambda v: "font-weight: bold;", axis="index", level=0)
    table_latex: str = table \
        .to_latex(
            column_format='rr'+'r'*len(df.columns),
            position="h",
            position_float="centering",
            hrules=True,
            label=f'table:results_rq2_summary',
            multirow_align="t",
            multicol_align="c",
            convert_css=True,
            caption='RQ2 - F-Measure variation according to Noise Level (%)'
        )

    table_latex = table_latex.replace('%', '\%')
    table_latex = table_latex.replace(
        "\\begin{tabular}", "\\tiny \n \\begin{tabular}")
    
    # change \midrule location
    lines = table_latex.split('\n')
    midrule_line = lines.index('\\midrule')
    lines[midrule_line] = ''
    lines.insert(midrule_line-1, '\\midrule')
    table_latex = '\n'.join(lines)

    with open(filename, 'w+') as f:
        f.write(table_latex)

    return table_latex, table

def gen_raw_values_table(df: pd.DataFrame, main_path, percent_noise):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    df = _prepare_table_to_latex(df, percent_noise)

    noise_column_name = 'Noise Level' + (' (%)' if percent_noise else ' (UN)')
    df = df.rename(columns={"Noise Level": noise_column_name})

    df.loc[:, noise_column_name] = pd.to_numeric(df[noise_column_name])

    group: pd.DataFrame
    for provider, group in df.groupby('Provider'):
        provider = provider.capitalize()
        group = group[["Noise Algorithm", noise_column_name, "F-Measure"]]

        group = group.set_index(['Noise Algorithm'])
        group = group.groupby('Noise Algorithm', group_keys=False).apply(
            lambda x: x.sort_values(by=[noise_column_name]))
        group = group.reset_index()

        group = divide_dataframe(group)

        filename = main_path + f'/table_latex_{provider}.txt'
        
        group = group.style \
        .set_properties(**{'font-size': '4px'} ) \
        .hide(axis=0) \
        .format(precision=2)

        table_latex = group.to_latex(
            column_format="rrr|rrr", position="h", position_float="centering",
            hrules=True, label=f'table:results_{provider}', caption=f'Results of {provider} provider',
            multirow_align="t", multicol_align="r", convert_css=True
        )

        # on duplicating columns to save space (divide_dataframe), is added __1 to prevent duplicated columns
        table_latex = table_latex.replace('__1', '')
        table_latex = table_latex.replace('%', '\%')

        table_latex = table_latex.replace(
            "\\begin{tabular}", "\\tiny \n \\begin{tabular}")

        with open(filename, 'w+') as f:
            f.write(table_latex)