import yaml

def load_config():
    with open('config.yaml') as file:
        return yaml.safe_load(file)