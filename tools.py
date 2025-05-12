import yaml

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_prompt_techniques(prompts_config):
    # return [technique.replace("-", " ").title() for technique in list(prompts_config["prompts"].keys())]
    return list(prompts_config["prompts"].keys())

def get_rag_techniques(rag_config):
    # return [rag.replace("-", " ").title() for rag in rag_config["rags"]]
    return rag_config["rags"]