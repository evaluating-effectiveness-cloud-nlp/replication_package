import pandas as pd
from pathlib import Path

def save_data_to_file(data, path, file_name):
    df = pd.DataFrame(data, columns =['review'])
    file_name = file_name+'.xlsx'

    Path(path).mkdir(parents=True, exist_ok=True)

    df.to_excel(path+'/'+file_name, 'data', index=False)

def return_sentence_similarity(a, b):
    a = a.capitalize()
    b = b.capitalize()
    size = len(a) if len(a) > len(b) else len(b)
    a = a.ljust(size)
    b = b.ljust(size)
    equals = 0
    for i in range(size):
        if(a[i]==b[i]):
            equals+=1
    
    print(f'equals:{equals}, size:{size}, diference:{size-equals}')
    return equals/size