from biocypher import BioCypher
import pandas as pd
from networkcommons._utils import network_from_df
import yaml

def update_config(file_path, changes):
    """
    Update the YAML configuration file with the provided changes.

    Args:
        file_path (str): Path to the YAML configuration file.
        changes (dict): Changes to apply to the configuration. Keys are dot-separated paths to the config values to update.

    Returns:
        None
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return

    for key, value in changes.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

# Example usage:
config_changes = {
    'biocypher.dbms': 'postgres',
    'biocypher.offline': False,
    'biocypher.debug': True,
    'biocypher.output_directory': 'new-output-directory'
}

update_config('path/to/your/config.yaml', config_changes)


def get_config_options(file_path):
    """
    Retrieve all available options from the YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration options.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return {}


def init_biocypher(dbms: Optional[str] = 'neo4j', 
                   offline: Optional[bool] = True,
                   strict_mode: Optional[bool] = False,
                   db_name: Optional[str] = None):
    bc = BioCypher(dbms=dbms,
                   offline=offline,
                   strict_mode=strict_mode,
                   db_name=db_name)
    
    return bc

def biocypher_to_graph(bc_obj):
    """
    Convert a BioCypher object to a networkx graph.

    Args:
        bc_obj (BioCypher): The BioCypher object to convert.

    Returns:
        nx.DiGraph: The networkx graph representation of the BioCypher object.
    """
    network_df = bc_obj.to_df()
    network = network_from_df(network_df,
                              source_col='source',
                              target_col='target',
                              directed=True)
    return network


from bccb.ppi_adapter import get_ppi_edges




