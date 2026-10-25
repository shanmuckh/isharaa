import json
import os

def _get_config_path():
    # Try local src/config.json, then sibling config.json in src dir
    if os.path.exists('src/config.json'):
        return 'src/config.json'
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(dir_path, 'config.json')
    if os.path.exists(candidate):
        return candidate
    return 'src/config.json'

def get_classes(): 
    default_classes = ["hello", "iloveyou", "one", "thankyou", "zero"]  # Label Studio id order
    try: 
        cfg_path = _get_config_path()
        with open(cfg_path) as f: 
            config = json.load(f) 
        classes = config.get('classes', default_classes)
        assert len(classes) > 0
        return classes         
    except Exception: 
        return default_classes

def get_colors(): 
    default_colors = [
        [41, 171, 202], 
        [240, 172, 95], 
        [131, 193, 103], 
        [235, 87, 87], 
        [155, 89, 182]
    ]
    try: 
        cfg_path = _get_config_path()
        with open(cfg_path) as f: 
            config = json.load(f) 
        classes = config.get('classes', [])
        colors = config.get('colors', default_colors)
        if len(classes) == len(colors) and len(colors) > 0:
            return colors
        return default_colors
    except Exception: 
        return default_colors
    
    
if __name__ == '__main__': 
    classes = get_classes()
    print(classes) 