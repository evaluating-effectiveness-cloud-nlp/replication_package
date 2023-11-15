import json
from pathlib import Path

def save_progress(main_path: str, progress_json):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    with open(main_path+'/progress.json', 'w') as f:
        json.dump(progress_json, f)

def load_progress(main_path: str):
    f = open(main_path+'/progress.json')
    data = json.load(f)
    f.close()

    return data

def init_progress(main_path, noise_algorithms, noise_levels, ml_providers, override=False):
    progress_file = main_path+'/progress.json'
    path = Path(progress_file)
    if not override and path.is_file():
        return load_progress(main_path)
    
    progress = {"noise":{}, "predictions":{}}
    
    noises ={}
    for n in noise_algorithms:
        noise_list = {}
        for l in noise_levels:
            noise_list[str(float(l))]=None
        noises[n.__name__]=dict(noise_list)

    progress["noise"]=dict(noises)
    
    providers = {}
    for p in ml_providers:
        providers[p.__name__]=dict(noises)
        providers[p.__name__]["no_noise"] = {"0.0": None}  
     
     
    progress["predictions"]=dict(providers)
    
    save_progress(main_path, progress)
    return progress