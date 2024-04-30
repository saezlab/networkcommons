import omnipath as op
import numpy as np


def get_omnipath(genesymbols=True,
                 directed_signed=True):
    """
    Retrieves the Omnipath network with directed interactions and specific criteria.

    Returns:
        network (pandas.DataFrame): Omnipath network with source, target, and sign columns.
    """
    network = op.interactions.AllInteractions.get('omnipath', genesymbols=genesymbols)

    network.rename(columns={'source':'source_uniprot', 'target':'target_uniprot'}, inplace=True)
    network.rename(columns={'source_genesymbol':'source', 'target_genesymbol':'target'}, inplace=True)

    # get only directed and signed interactions
    if directed_signed:
        network = network[(network['consensus_direction'] == True) & 
                          (((network['consensus_stimulation'] == True) & (network['consensus_inhibition'] == False)) | 
                           ((network['consensus_stimulation'] == False) & (network['consensus_inhibition'] == True))) & 
                           (network['curation_effort'] >= 2)]

    network['sign'] = np.where(network['consensus_stimulation'] == True, 1, -1)

    # write the resulting omnipath network in networkx format
    network = network[['source', 'target', 'sign']].reset_index(drop=True)

    return network