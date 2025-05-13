import os
from typing import List

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

def create_files(files) -> List[str]:
    data_dir = "data"
    saved_paths = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if files:
        for file in files:
            file_path = os.path.join(data_dir, file.name)
            saved_paths.append(file_path)

            try:
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
            except Exception as e:
                print(e)

    return saved_paths