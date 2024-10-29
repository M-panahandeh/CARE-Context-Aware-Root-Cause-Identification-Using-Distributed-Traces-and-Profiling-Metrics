import psutil
import time
import json
import math
import multiprocessing
#from infomap import infomap
import infomap
import pandas as pd
from collections import defaultdict
from colorama import Back, Fore, Style, init
from tkinter import Tk, filedialog
from tkinter.filedialog import askopenfilename
import pickle
import numpy as np
import os
import networkx as nx
from itertools import groupby

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('TkAgg')  # Choose an appropriate backend
import matplotlib.pyplot as plt
import community
from collections import defaultdict
import torch
from torch_geometric.utils import from_networkx
import torch.optim as optim
from GNN import GraphSAGE,train_unsupervised,cluster_embeddings_spectral,evaluate_clustering,calculate_similarity_matrix

def process_file(file_path):
    print(f"Processing {file_path} started.")  # Debugging information
    # Read the contents of the pickle file
    file_name = os.path.basename(file_path)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    tic = time.time()
    # Get current process ID (PID)
    pid = os.getpid()
    process = psutil.Process(pid)
    # root_cause = {file_name.split('_')[0]}
    root_cause = f"ts-{file_name.split('_')[0]}-service"
    # root_cause = f"{file_name.split('_')[0]}" #for API and container
    abnormal_traces = []
    normal_traces = []
    predict_trace_ids = data.loc[data['predict'] == 1.0, 'trace_id'].unique()
    all_abnormal_trace_ids=predict_trace_ids
    abnormal_traces_df = data[data['trace_id'].isin(all_abnormal_trace_ids)]
    # convert to a list of dictionaries:
    abnormal_traces = abnormal_traces_df.to_dict(orient='records')
    # Filter data where trace_id is not in unique_trace_ids
    normal_traces_df = data[~data['trace_id'].isin(all_abnormal_trace_ids)]
    # convert to a list of dictionaries:
    normal_traces = normal_traces_df.to_dict(orient='records')

    #################################################Step 1: Create call graphs nt and at ####################################################
    # Create at_graph call graph= call graph extracted from abnormal traces
    at_graph = generate_graph(abnormal_traces)
    # Draw the graph
    # draw_graph(at_graph)
    # Create nt_graph call graph= call graph extracted from normal traces
    nt_graph = generate_graph(normal_traces)
    # Draw the graph
    # draw_graph(nt_graph)
    ###################################################Step 2: Apply community-based algorithm########################################
    # Apply clustering algorithm to detect communities in at_graph
    at_communities = GNNCummunity(at_graph)
    #print(communities)
    nt_communities = GNNCummunity(nt_graph)

    # To visualize the communities by colors
    # Set up a list of colors for each community
    # at_colors = [at_communities[node] for node in at_graph.nodes()]
    # nt_colors = [nt_communities[node] for node in nt_graph.nodes()]
    # #draw comminuties with colors
    # draw_communiti_based_graphs(at_graph,at_colors)
    # draw_communiti_based_graphs(nt_graph, nt_colors)
   #####################################################Step 3: Apply PageRank based on communities#####################################################
    # Create a list to store the subgraphs for each community in at_graph
    at_node_scores=global_combinational_Page_rank_Louvain_GNN(at_graph,at_communities)
    #for normal traces
    nt_node_scores= global_combinational_Page_rank_Louvain_GNN(nt_graph,nt_communities)
    #without community
    # at_node_scores = PageRank_withoutCommunity(at_graph)
    # # for normal traces
    # nt_node_scores = PageRank_withoutCommunity(nt_graph)

    ###########################################Step 4: Apply Weighted Spectrum Analysis#######################################
    # Create a dictionary to store the count for each node in at_graph
    at_trace_order=heuristic_trace_priorization(abnormal_traces, at_graph, at_node_scores)
    nt_trace_order= heuristic_trace_priorization(normal_traces, nt_graph, nt_node_scores)

    # #no path analysis
    # O_ef, O_nf = count_services(abnormal_traces, at_graph)
    # O_ep, O_np = count_services(normal_traces, nt_graph)
    #with path analysis
    O_ef, O_nf = weighted_count_services(at_graph,at_trace_order,abnormal_traces)
    O_ep, O_np = weighted_count_services(nt_graph,nt_trace_order,normal_traces)


    # Weighted notations for O_ef and O_nf:
    O_ef_weighted = {}
    O_nf_weighted = {}
    O_ep_weighted = {}
    O_np_weighted = {}
    for node in at_graph.nodes():
        O_ef_weighted[node] = O_ef.get(node, 0) * at_node_scores.get(node, 1)
        O_nf_weighted[node] = O_nf.get(node, 0) * at_node_scores.get(node, 1)
    for node in nt_graph.nodes():
        O_ep_weighted[node] = O_ep.get(node, 0) * nt_node_scores.get(node, 1)
        O_np_weighted[node] = O_np.get(node, 0) * nt_node_scores.get(node, 1)

    ############################################ Step 5: Output ranked list############################################

    list_of_candiates = {}
    for node in set(at_graph.nodes()) | set(nt_graph.nodes()):
        try:
         # #ochiai
            list_of_candiates[node] = (O_ef_weighted.get(node, 0)) / math.sqrt(
                (O_ef_weighted.get(node, 0) + O_ep_weighted.get(node, 0)) * (
                        O_ef_weighted.get(node, 0) + O_nf_weighted.get(node, 0)))


        # Dstar2
        #       list_of_candiates[node] =(O_ef_weighted.get(node, 0)**2.0)/(O_ep_weighted.get(node, 0)+O_nf_weighted.get(node, 0))


        # RussellRao
        #         list_of_candiates[node] =O_ef_weighted.get(node, 0)/(O_ef_weighted.get(node, 0)+O_nf_weighted.get(node, 0)+O_ep_weighted.get(node, 0)+O_np_weighted.get(node, 0))

        # M2
        #           list_of_candiates[node] = O_ef_weighted.get(node, 0)/(O_ef_weighted.get(node, 0)+O_np_weighted.get(node, 0)+(2*O_ep_weighted.get(node, 0))+(2*O_nf_weighted.get(node, 0)))

        # list_of_candiates
        # list_of_candiates[service] = (O_ef.get(service, 0)) / math.sqrt(
        #     (O_ef.get(service, 0) + O_ep.get(service, 0)) * (
        #             O_ef.get(service, 0) + O_nf.get(service, 0)))
        except ZeroDivisionError:
            list_of_candiates[node] = 0

    toc = time.time()
    # Measure memory usage
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
    # Measure CPU usage
    cpu_percent = psutil.cpu_percent()

    # Sort services based on their Ochiai scores (high to low)
    sorted_list_of_candiates = sorted(list_of_candiates.items(), key=lambda x: x[1], reverse=True)
    # Iterate through the sorted_ochiai and assign ranks
    rank = 0
    prev_score = None
    ranked_list_of_candiates = {}
    for service, score in sorted_list_of_candiates:
        # If the score is same as the previous score, keep the same rank
        if prev_score == score:
            ranked_list_of_candiates[service] = (score, rank)
        else:
            rank=rank+1
            ranked_list_of_candiates[service] = (score, rank)

        prev_score = score

    init(autoreset=True)
    print(Back.GREEN + "Ranked list:\n")
    for service, (score, rank) in ranked_list_of_candiates.items():
        print(f"{rank}\t{service}\t{score}")

    print(f"Processing {file_path} completed.")  # Debugging information
    ################################################Step 6: find the rank of identified root cause########################
    return root_cause,ranked_list_of_candiates,(toc-tic),memory_usage,cpu_percent  #ranked_ochiai includes service: (score , rank)



def generate_graph(traces):
    try:
        graph = nx.DiGraph()
        # Iterate over each data entry in the dataset
        for entry in traces:
            trace_id = entry['trace_id']
            source = entry['source']
            target = entry['target']
            alpha_product = entry['alpha_product']
            # Add nodes if they don't exist
            if not graph.has_node(source):
                graph.add_node(source)
            if not graph.has_node(target):
                graph.add_node(target)

            if graph.has_edge(source, target):
                # If the edge exists, update its weight
                graph[source][target]['weight'] += alpha_product
            else:
                # If the edge does not exist, create it with the given weight
                graph.add_edge(source, target, weight=alpha_product)
        return graph
    except Exception as e:
        # Handle any exceptions that may occur during processing
        print(f"An exception occurred: {e}")

def draw_graph(graph):
    pos = nx.spring_layout(graph, seed=42)
    edge_labels = {(source, target): weight for source, target, weight in graph.edges(data='weight', default=1)}
    nx.draw(graph, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10,
             font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    # # Show the plot
    plt.show()

def apply_infomap(graph):
    try:
        # Check if graph is None
        if graph is None or not graph.nodes():
            return None
        # Initialize Infomap
        im = infomap.Infomap("-d -s 1234")  # -d for directed graphs and -s for random seed for reproducibility

        # Build Infomap network from the NetworkX graph
        # Create a mapping from node identifiers to integer indices
        node_to_index = {node: index for index, node in enumerate(graph.nodes())}
        # Create a reverse mapping from integer indices to node identifiers
        index_to_node = {index: node for node, index in node_to_index.items()}
        for e in graph.edges(data=True):
            im.addLink(node_to_index[e[0]], node_to_index[e[1]], float(e[2]['weight']))
        # Run Infomap
        im.run()
        # Extract communities
        communities = {}
        # Get modules at the finest level (leaf nodes)
        modules_at_finest_level = im.get_modules(depth_level=-1)
        # Create a mapping from node names to their corresponding module IDs
        for node_id, module_id in modules_at_finest_level.items():
            communities[index_to_node[node_id]] = module_id
        return communities
    except Exception as e:
        # Handle any exceptions that may occur during processing
        print(f"An exception occurred: {e}")

def draw_communiti_based_graphs(graph,colors):
    # Draw the at_graph with communities
    pos = nx.spring_layout(graph, seed=42)
    edge_labels = {(source, target): weight for source, target, weight in graph.edges(data='weight', default=1)}
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, node_color=colors, cmap=plt.get_cmap('viridis'), node_size=1000,
            with_labels=True, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Communities ")
    plt.show()
def apply_Louvian(graph):
    if graph is None or not graph.nodes():
        return None
    # Convert the directed graphs to non-directed graphs
    graph_undirected = graph.to_undirected()
    # Calculate the modularity based on directed edges
    # communities = community.best_partition(graph, weight='weight', directed=True)
    communities = community.best_partition(graph_undirected)
    # Create a dictionary to store the nodes belonging to each community in at_graph
    communities_dict = {}
    for node, community_id in communities.items():
        if community_id not in communities_dict:
            communities_dict[community_id] = [node]
        else:
            communities_dict[community_id].append(node)
    return communities_dict

def find_weighted_groups(graph, weight_threshold=1): #heuristic for community detection
    # Step 1: Edge Weight Analysis
    outgoing_weight_sums = {node: sum(data['weight'] for _, _, data in graph.out_edges(node, data=True)) for node in graph}
    sorted_nodes = sorted(outgoing_weight_sums.items(), key=lambda x: -x[1])

    # Step 2: Graph Traversal for Grouping
    groups = defaultdict(lambda: {'nodes': [], 'raw_score': 0, 'normalized_score': 0})
    current_group = 0
    added_nodes = set()
    for node, weight in sorted_nodes:
        if node not in added_nodes:
            groups[current_group]['raw_score'] += weight
            queue = [node]
            while queue:
                current_node = queue.pop(0)
                if current_node not in added_nodes:
                    groups[current_group]['nodes'].append(current_node)
                    added_nodes.add(current_node)
                    for _, neighbor, data in graph.out_edges(current_node, data=True):
                        if data['weight'] > weight_threshold and neighbor not in added_nodes:
                            queue.append(neighbor)
                            groups[current_group]['raw_score'] += data['weight']
            num_nodes = len(groups[current_group]['nodes'])
            groups[current_group]['normalized_score'] = groups[current_group]['raw_score'] / num_nodes if num_nodes > 0 else 0
            current_group += 1

    return dict(groups)


def GNNCummunity(graph):
    G=graph
    # Create a mapping from node ID to node name
    node_id_to_name = {i: node for i, node in enumerate(G.nodes())}
    #convert graph to pytorch geometric format
    for node in G.nodes():
        degree = G.degree(node)
        outbound_weight = sum(data['weight'] for _, _, data in G.out_edges(node, data=True))
        inbound_weight = sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
        G.nodes[node]['feature'] = [degree, outbound_weight, inbound_weight]

    data = from_networkx(G)
    data.x = torch.tensor([G.nodes[node]['feature'] for node in G.nodes()], dtype=torch.float)
    data.edge_weight = torch.tensor([data['weight'] for _, _, data in G.edges(data=True)], dtype=torch.float)

    # Normalize features and edge weights
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    data.edge_weight = (data.edge_weight - data.edge_weight.mean()) / data.edge_weight.std()

    # Initialize the model and optimizer
    #in_channels=3: This specifies the number of input features.  each node has three features: degree, outbound weight, and inbound weight.
    #hidden_channels=16: This specifies the number of hidden units (neurons) in the first layer of the GraphSAGE model.
    # out_channels=2: This specifies the number of output features for each node.
    model = GraphSAGE(in_channels=3, hidden_channels=16, out_channels=2)  # Adjust in_channels to match the feature size
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    model = train_unsupervised(data, model, optimizer, epochs=200)
    # Get the embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
    # Cluster the embeddings using Spectral Clustering
    #number f clusters
    # clusters = cluster_embeddings_spectral(embeddings, max_cluster=G.number_of_nodes(), num_clusters=None)
    if G.number_of_nodes()<10:
        max_cluster = G.number_of_nodes()
    else:
        max_cluster= 10
    # max_cluster= G.number_of_nodes()/2
    clusters = cluster_embeddings_spectral(embeddings, max_cluster, num_clusters=None)
    # Evaluate the clustering
    similarity_matrix=calculate_similarity_matrix(embeddings)

    evaluate_clustering(similarity_matrix, clusters)

    # Create a dictionary to store the communities
    # Create a dictionary to store the communities
    communities = defaultdict(list)
    for node_id, cluster in enumerate(clusters):
        node_name = node_id_to_name[node_id]
        communities[cluster].append(node_name)
    return dict(communities)


def score_nodes(graph,communities):
    # Calculate PageRank with a global approch (community rank* node rank in the general graph), developed for huristic cummunity detection (find_weighted_groups)
    pagerank_scores = nx.pagerank(graph, weight='weight')
    # Initialize node scores
    node_scores = {}
    # For each group, calculate the score for each node
    for group_id, group_info in communities.items():
        # group_score = group_info['normalized_score']
        group_score = group_info['raw_score'] #works better
        for node in group_info['nodes']:
            node_scores[node] = pagerank_scores[node] * group_score
    return node_scores

#calculate node score in the general graph * PR claculated for graph of communites. developed for huristic community detection
def global_combinational_Page_rank2(graph, communities):
    # Check if either graph or communities is None or empty
    if graph is None or not graph.nodes() or communities is None or not communities:
        return None  # or raise an error or return some default value
    # Initialize node scores
    node_scores = {}
    pagerank_scores = nx.pagerank(graph, weight='weight')
    # Construct a new graph where each community is a node.
    community_graph = nx.Graph()
    for group_id, group_info in communities.items():
        community_graph.add_node(group_id)
    # Connect these "community nodes" based on interactions between their member services
    for source, target, data in graph.edges(data=True):
        try:
            weight = data['weight']
            source_community = None  # Initialize source_community
            target_community = None  # Initialize target_community
            # Find the community ID for the source node
            for community_id, community_info in communities.items():
                if source in community_info['nodes']:
                    source_community = community_id
                    break  # Stop searching once the community is found
            # Find the community ID for the target node
            for community_id, community_info in communities.items():
                if target in community_info['nodes']:
                    target_community = community_id
                    break  # Stop searching once the community is found
            # Check if both source and target nodes belong to communities
            if source_community is not None and target_community is not None:
                if community_graph.has_edge(source_community, target_community):
                    community_graph[source_community][target_community]['weight'] += weight
                else:
                    community_graph.add_edge(source_community, target_community, weight=weight)
            else:
                print("Error: One or both nodes do not belong to any community.")
        except KeyError as e:
            print(f"Error processing edge: {e}")

    # PageRank on communities as nodes
    community_rankings = nx.pagerank(community_graph, weight='weight',alpha=0.85)

    # For each group, calculate the score for each node
    for group_id, group_info in communities.items():
        # group_score = group_info['normalized_score']
        # group_score=community_rankings[group_id]+group_info['raw_score']
        group_score = community_rankings[group_id]
        for node in group_info['nodes']:
            #different node score combinations
            # node_scores[node] = pagerank_scores[node] * group_score
            # node_scores[node] = (pagerank_scores[node] + group_score)/2
            node_scores[node] =( (0.2*pagerank_scores[node]) + (0.8*group_score) )/ 2
    return node_scores

def local_combinational_Page_rank3(graph, communities):
    node_scores={}
    #calculate node pagerank inside of communities*community score. developed for huristic communities
    # Create an empty list to store community subgraphs
    community_subgraphs = []
    # Iterate over community IDs and extract nodes for each community
    for community_info in communities.values():
        community_nodes = community_info['nodes']
        # Create a subgraph containing only the nodes belonging to the current community
        community_subgraph = graph.subgraph(community_nodes)
        community_subgraphs.append(community_subgraph)
    nodes_pagerank_results = [nx.pagerank(subgraph) for subgraph in community_subgraphs]

    # Construct a new graph where each community is a node.
    community_graph = nx.Graph()
    for group_id, group_info in communities.items():
        community_graph.add_node(group_id)
    # Connect these "community nodes" based on interactions between their member services
    for source, target, data in graph.edges(data=True):
        try:
            weight = data['weight']
            source_community = None  # Initialize source_community
            target_community = None  # Initialize target_community

            # Find the community ID for the source node
            for community_id, community_info in communities.items():
                if source in community_info['nodes']:
                    source_community = community_id
                    break  # Stop searching once the community is found

            # Find the community ID for the target node
            for community_id, community_info in communities.items():
                if target in community_info['nodes']:
                    target_community = community_id
                    break  # Stop searching once the community is found

            # Check if both source and target nodes belong to communities
            if source_community is not None and target_community is not None:
                if community_graph.has_edge(source_community, target_community):
                    community_graph[source_community][target_community]['weight'] += weight
                else:
                    community_graph.add_edge(source_community, target_community, weight=weight)
            else:
                print("Error: One or both nodes do not belong to any community.")

        except KeyError as e:
            print(f"Error processing edge: {e}")

    # PageRank on communities as nodes
    community_rankings = nx.pagerank(community_graph, alpha=0.85)

    # For each group, calculate the score for each node
    for group_id, group_info in communities.items():
        # group_score = group_info['normalized_score']
        # group_score = community_rankings[group_id] + group_info['raw_score']
        group_score = community_rankings[group_id]
        for node in group_info['nodes']:
            pagerank_dict = nodes_pagerank_results[group_id]
            pagerank_score = pagerank_dict.get(node, None)
            node_scores[node] = pagerank_score * group_score
    return node_scores

def global_combinational_Page_rank(graph,communities):
    # Check if either graph or communities is None or empty
    if graph is None or not graph.nodes() or communities is None or not communities:
        return None  # or raise an error or return some default value
    #Construct a new graph where each community is a node.
    community_graph = nx.Graph()
    for community, nodes in communities.items():
        community_graph.add_node(community)
    #Connect these "community nodes" based on interactions between their member services
    for source, target, data in graph.edges(data=True):
      try:
        source_community = communities[source]
        target_community = communities[target]
        weight = data['weight']
        #The weight is cumulative interactions between members of two communities.
        if community_graph.has_edge(source_community, target_community):
            community_graph[source_community][target_community]['weight'] += weight
        else:
            community_graph.add_edge(source_community, target_community, weight=weight)
      except KeyError as e:
          print(f"Error processing edge: {e}")

    #page rank on communities as nodes
    community_rankings = nx.pagerank(community_graph, alpha=0.85)

    #compute PageRank in general graph
    node_rankings = nx.pagerank(graph, alpha=0.85, weight='modified_weight')
    # # Eigenvector centrality on communities as nodes
    # community_rankings = nx.eigenvector_centrality(community_graph)
    # # Compute Eigenvector centrality in the general graph
    # node_rankings = nx.eigenvector_centrality(graph, weight='modified_weight')
    # community_rankings=nx.betweenness_centrality(community_graph)
    # node_rankings=nx.betweenness_centrality(graph, weight='modified_weight')
    #combine ranks:
    combined_rankings = {}
    for node, rank in node_rankings.items():
        community = communities[node]
        combined_rankings[node] = rank * community_rankings[community]
    #sort
    #sorted_nodes = sorted(combined_rankings.keys(), key=lambda x: combined_rankings[x], reverse=True)
    return combined_rankings

def global_combinational_Page_rank_Louvain_GNN(graph, communities):
    #this is for louvian
    # Check if either graph or nt_communities is None or empty
    if graph is None or not graph.nodes() or communities is None or not communities:
        return None  # or raise an error or return some default value
    # Construct a new graph where each community is a node.
    community_graph = nx.Graph()
    for community, nodes in communities.items():
        community_graph.add_node(community)
    # Connect these "community nodes" based on interactions between their member services
    for source, target, data in graph.edges(data=True):
        try:
            source_community = [community for community, nodes in communities.items() if source in nodes][0]
            target_community = [community for community, nodes in communities.items() if target in nodes][0]
            weight = data['weight']
            # The weight is cumulative interactions between members of two communities.
            if community_graph.has_edge(source_community, target_community):
                community_graph[source_community][target_community]['weight'] += weight
            else:
                community_graph.add_edge(source_community, target_community, weight=weight)
        except IndexError:
            print(f"Error processing edge: {source}, {target}")
    # PageRank on communities as nodes
    community_rankings = nx.pagerank(community_graph, alpha=0.85)
    # Compute PageRank in the general graph
    node_rankings = nx.pagerank(graph, alpha=0.85, weight='modified_weight')
    # Combine ranks:
    combined_rankings = {}
    for node, rank in node_rankings.items():
        community = [community for community, nodes in communities.items() if node in nodes][0]
        combined_rankings[node] = rank * community_rankings[community]
    return combined_rankings


def localPageRank(graph,communities_dict):
    #rank nodes inside of communities
    community_subgraphs = [graph.subgraph(community_id) for community_id in
                              communities_dict.values()]
    # Calculate PageRank for each community subgraph in at_graph
    pagerank_results = [nx.pagerank(subgraph) for subgraph in community_subgraphs]
    # Print PageRank scores for at_graph
    print("PageRank scores for graph:")
    for i, community_nodes in enumerate(communities_dict.values()):
        print(f"Community {i + 1}:")
        pagerank_scores = pagerank_results[i]
        for node, score in pagerank_scores.items():
            print(f"Node: {node}, PageRank Score: {score}")
        print()
    return pagerank_scores


def localPageRank_infomap(graph, communities_dict):
    #this is for informap
    # Initialize a dictionary to store PageRank scores for each node
    pagerank_scores = {}
    # Iterate through communities and calculate PageRank scores for each community subgraph
    for node, community_id in communities_dict.items():
        # Create a subgraph for the nodes in the current community
        community_nodes = [n for n, cid in communities_dict.items() if cid == community_id]
        community_subgraph = graph.subgraph(community_nodes)

        # Calculate PageRank for the community subgraph
        community_pagerank = nx.pagerank(community_subgraph)

        # Store the PageRank scores in the pagerank_scores dictionary
        pagerank_scores.update(community_pagerank)
    # Print PageRank scores
    print("PageRank scores for communities:")
    for node, score in pagerank_scores.items():
        print(f"Node: {node}, PageRank Score: {score}")
    return pagerank_scores

def PageRank_withoutCommunity(graph):
    #without community detection
    node_rankings = nx.pagerank(graph, alpha=0.85, weight='modified_weight')
    # Convert node rankings to use names instead of IDs
    return node_rankings

def count_services(traces,graph): #this count spectrum notation without considering path
    O_e = {}
    O_n = {}
    for service in graph.nodes():
        # Find trace_ids where the service is either a source or a target
        relevant_traces = [trace['trace_id'] for trace in traces if
                           trace['source'] == service or trace['target'] == service]
        len_relevent = len(set(relevant_traces))
        # Find trace_ids where the service is neither a source nor a target
        non_relevant_traces = [trace['trace_id'] for trace in traces if
                               trace['source'] != service and trace['target'] != service]
        len_nonrelevant = len(set(non_relevant_traces))

        O_e[service] = len_relevent
        O_n[service] = len_nonrelevant
    return O_e,O_n
################Path analysis##################################################
def traces_score_based_node(traces,graph, node_scores): #prioritize traces based on their importacne and diversity
    # Collect Sources and Targets for each trace_id
    trace_sequences = defaultdict(list)
    for trace in traces:
        trace_sequences[trace['trace_id']].append((trace['source'], trace['target']))
    # cluster traces with the same type: Find Unique Sequences and assign cluster IDs
    unique_sequences = {}
    for sequences in trace_sequences.values():
        sequences_tuple = tuple(sequences)
        unique_sequences.setdefault(sequences_tuple, len(unique_sequences))
    #Cluster Traces
    traces_cluster = {}
    for trace_id, sequences in trace_sequences.items():
        sequence_key = unique_sequences[tuple(sequences)]
        if sequence_key not in traces_cluster:
            traces_cluster[sequence_key] = {'trace_ids': [], 'sequence': sequences}
        traces_cluster[sequence_key]['trace_ids'].append(trace_id)

    #Calculate the node score for each cluster
    # Sort nodes based on scores
    sorted_nodes = sorted(graph.nodes(), key=lambda node: node_scores.get(node, 0))
    node_positions = {node: pos for pos, node in enumerate(sorted_nodes, start=1)}
    total_score_sum = sum(node_scores.values())

    # Calculate trace scores for each cluster
    cluster_scores = {}
    for cluster_id, traces in traces_cluster.items():
        # Find the highest position and calculate mean score of nodes in the cluster
        highest_position = -1
        score_sum = 0
        node_count = 0
        trace_type= traces['sequence']
        unique_nodes = set() #to only study unique nodes within a trace
        for source, target in trace_type:
                unique_nodes.add(source)
                unique_nodes.add(target)
        for node in unique_nodes:
                if node in node_scores:
                    position = node_positions[node]
                    highest_position = max(highest_position, position)
                    score_sum += node_scores[node]
                    node_count += 1
        mean_score = score_sum / node_count if node_count else 0
        trace_score = ((highest_position - 1) / len(graph.nodes())) + (
                    mean_score / (len(graph.nodes()) * total_score_sum))
        cluster_scores[cluster_id] = trace_score
    return cluster_scores,traces_cluster

def heuristic_trace_priorization(traces, graph, node_scores):
    cluster_scores, traces_cluster = traces_score_based_node(traces, graph, node_scores)
    normalized_cluster_scores = normalize_scores(cluster_scores)# to make it comparative with diversity

    #compute diversity and score huristicly
    order = {}
    remaining_clusters = traces_cluster.copy()  # Copy to avoid modifying the original traces_cluster

    while remaining_clusters:
        prev = list(order.keys())[-1] if order else None #get the last added id to order
        comp = {}
        for cls_id, cls_info in remaining_clusters.items():
            comp1 = normalized_cluster_scores[cls_id]
            if prev is not None:
                comp1 -= calculate_jaccard_distance_for_clusters(cls_id, prev, traces_cluster)
            comp[cls_id] = comp1

        max_cluster_id = max(comp, key=comp.get)  # Find the id for the max computed one
        order[max_cluster_id] = {
            'sequence': tuple(remaining_clusters[max_cluster_id]['sequence']),
            'score': comp[max_cluster_id]
        }
        # Note: del removes the key, and the indexes are not updated since it's a dictionary
        del remaining_clusters[max_cluster_id]
    # return order,traces_cluster #list of cluster ids and their combined score

    #extract alpha-value for each trace
    dict_traces_alpha=traces_score_based_alpha(traces)
    result_dict = {}
    # Create a dictionary to map trace_id to cluster_id
    trace_id_to_cluster = {trace_id: cluster_id for cluster_id, value in traces_cluster.items() for trace_id in
                           value['trace_ids']}

    #for each trace-id find the value based on dict_traces and order
    for trace in traces:
        trace_id = trace['trace_id']
        # Get the sum of alpha_product from result_dict
        alpha_product_value = dict_traces_alpha.get(trace_id, 0)  # Default to 0 if trace_id is not in result_dict
        # Get the cluster_id from the pre-built mapping
        cluster_id = trace_id_to_cluster.get(trace_id)
        # Get the score from the order dictionary using the cluster_id
        order_value = order.get(cluster_id, {}).get('score', 0) if cluster_id is not None else 0
        # Calculate the final value for the trace_id
        result_value = order_value + alpha_product_value
        # result_value = order_value
        # result_value =  alpha_product_value
        # Store the result in the dictionary
        result_dict[trace_id] = result_value

    return result_dict #a dictionary of trace_id and value: node importance,diversity, and alpha value for the path



def weighted_count_services(graph,traces_score,traces): #this count spectrum notation considering path
    O_e = {}
    O_n = {}
    for service in graph.nodes():
        unique_trace_ids_e = set()
        unique_trace_ids_n = set()

        for trace in traces:
            if trace['source'] == service or trace['target'] == service:
                unique_trace_ids_e.add(trace['trace_id'])
            else:
                unique_trace_ids_n.add(trace['trace_id'])

        O_e[service] = sum(traces_score[trace_id] for trace_id in unique_trace_ids_e)
        O_n[service] = sum(traces_score[trace_id] for trace_id in unique_trace_ids_n)

    return O_e,O_n

def normalize_scores(scores):
    num_items = len(scores)
    if (min(scores.values())==max(scores.values())): #one cluster or clusters with similar sequences
        # If only one item, return a normalized score of 1.0
        return {k: 1.0 for k, v in scores.items()}
    else:
        min_score = min(scores.values())
        max_score = max(scores.values())
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}


def calculate_jaccard_distance(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return 1 -( len(intersection) / len(union))

def calculate_jaccard_distance_for_clusters(cluster_a, cluster_b, traces_cluster):
    set_a = {node for pair in traces_cluster[cluster_a]['sequence'] for node in pair}
    set_b = {node for pair in traces_cluster[cluster_b]['sequence'] for node in pair}
    return calculate_jaccard_distance(set_a, set_b)

def traces_score_based_alpha(traces):
    # Grouping by trace_id
    grouped_traces = groupby(traces, key=lambda x: x['trace_id'])

    # Calculating the sum of alpha_product for entries with the same trace_id
    # result_dict = {key: sum(item['alpha_product'] for item in group) for key, group in grouped_traces}

    result_dict = {}
    for key, group in grouped_traces:
        group_list = list(group)
        total_alpha_product = sum(item['alpha_product'] for item in group_list)
        group_length = len(group_list)
        average_alpha_product = total_alpha_product / group_length if group_length != 0 else 0
        result_dict[key] = average_alpha_product

    return result_dict




if __name__ == '__main__':

        count = 0  # number of files
        sum=0
        similar_position=0
        total_time=0
        total_memory = 0
        total_cpu = 0

        folder_path = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\NewLabeld_alpha_final"
        # folder_path = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\Debug"
        file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

        # Create a DataFrame
        df = pd.DataFrame(columns=['Filename', 'Rank', 'Score', 'Number of Services with Equal Ranks'])
        # Create a ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Submit the process_file tasks to the executor
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
            #process results
            for future in concurrent.futures.as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    root_cause, ranked_ochiai ,time, memory,cpu= future.result() #ranked_ochiai is service:score ,rank
                    if root_cause in ranked_ochiai:
                        # get rank of root cause
                        score, rank = ranked_ochiai.get(root_cause, (None, None))
                        # services with the same rank
                        services_with_same_rank = [service for service, (_, r) in ranked_ochiai.items() if
                                                   r == rank and service != root_cause]
                        #write result in excel
                        df_temp = pd.DataFrame({
                            'Filename': [file_name],
                            'Rank': [rank],
                            'Score': [score],
                            'Number of Services with Equal Ranks': [len(services_with_same_rank)]
                        })
                        # Concatenate the current DataFrame with the result DataFrame
                        df = pd.concat([df, df_temp], ignore_index=True)
                        #calculate averages
                        sum+=rank
                        count+=1 #process a file
                        similar_position+=len(services_with_same_rank)

                        total_time+=time
                        total_memory+=memory
                        total_cpu+=cpu

                except Exception as exc:
                    print(f'Error processing {file_name}: {exc}')
        df.to_excel('GNN.xlsx', index=False)
        print("\nResult:")
        print(f"Average rank: {(sum / count):.2f}")
        print(f"Average similar_position: {(similar_position / count):.2f}")

        print("totaltime:", total_time)
        print("memory:", memory)
        print("cpu:", cpu)








