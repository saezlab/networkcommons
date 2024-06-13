import omnipath as op
import numpy as np
import liana


def get_omnipath(genesymbols=True,
                 directed_signed=True):
    """
    Retrieves the Omnipath network with directed interactions
    and specific criteria.

    Returns:
        network (pandas.DataFrame): Omnipath network with
        source, target, and sign columns.
    """
    network = op.interactions.AllInteractions.get('omnipath',
                                                  genesymbols=genesymbols)

    network.rename(columns={'source': 'source_uniprot',
                            'target': 'target_uniprot'},
                   inplace=True)
    network.rename(columns={'source_genesymbol': 'source',
                            'target_genesymbol': 'target'},
                   inplace=True)

    # get only directed and signed interactions, with curation effort >2
    network = network[network['curation_effort'] >= 2]

    if directed_signed:
        network = network[(network['consensus_direction']) &
                          (
                              (network['consensus_stimulation']) |
                              (network['consensus_inhibition'])
                          )]

    network['sign'] = np.where(network['consensus_stimulation'], 1, -1)

    # write the resulting omnipath network in networkx format
    network = network[['source', 'target', 'sign']].reset_index(drop=True)

    return network


def get_lianaplus(resource='Consensus'):
    """
    Retrieves the Liana+ ligand-receptor interaction network with
    directed interactions and specific criteria.

    Args:
        resource (str, optional): The resource to retrieve the network from.
            Defaults to 'Consensus'.

    Returns:
        pandas.DataFrame: Liana+ network with source, target, and sign columns.
    """
    network = liana.resource.select_resource(resource).drop_duplicates()
    network.columns = ['source', 'target']
    network['sign'] = 1

    return network


# def build_moon_regulons(include_liana=False):

#     dorothea_df = dc.get_collectri()

#     TFs = np.unique(dorothea_df['source'])

#     full_pkn = get_omnipath(genesymbols=True, directed_signed=True)

#     if include_liana:
#         ligrec_resource = get_lianaplus()

#         full_pkn = pd.concat([full_pkn, ligrec_resource])
#         full_pkn['edgeID'] = full_pkn['source'] + '_' + full_pkn['target']

#         # This prioritises edges coming from OP
#         full_pkn = full_pkn.drop_duplicates(subset='edgeID')
#         full_pkn = full_pkn.drop(columns='edgeID')

#     kinTF_regulons = full_pkn[full_pkn['target'].isin(TFs)].copy()
#     kinTF_regulons.columns = ['source', 'target', 'mor']
#     kinTF_regulons = kinTF_regulons.drop_duplicates()

#     kinTF_regulons = kinTF_regulons.groupby(['source', 'target']).mean() \
#         .reset_index()

#     layer_2 = {}
#     activation_pkn = full_pkn[full_pkn['sign'] == 1].copy()

#     pkn_graph = network_from_df(activation_pkn, directed=True)

#     relevant_nodes = list(activation_pkn['source'].unique())
#     relevant_nodes = [node for node in relevant_nodes if node in list(kinTF_regulons['source'])]

#     for node in relevant_nodes:
#         intermediates = activation_pkn[activation_pkn['source'] == node]['target'].tolist()
#         targets = [n for i in intermediates for n in pkn_graph.neighbors(i)]
#         targets = np.unique([n for n in targets if n in TFs])
#         layer_2[node] = targets

#     layer_2_df = pd.concat([pd.DataFrame({'source': k, 'target': v, 'mor': 0.25}) for k, v in layer_2.items()], ignore_index=True)
#     kinTF_regulons = pd.concat([kinTF_regulons, layer_2_df])
#     kinTF_regulons = kinTF_regulons.groupby(['source', 'target']).sum().reset_index()

#     return kinTF_regulons


