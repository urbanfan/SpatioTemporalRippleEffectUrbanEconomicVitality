import arcpy
import os
import csv
import numpy as np
from scipy import signal
from datetime import datetime, timedelta
import math
import bisect
from typing import List, Dict, Tuple, Set
import time
import networkx as nx
from networkx.algorithms.community import modularity


class UniversalTreeSplitterGA:
    """
    Improved Universal Tree Splitter with Hierarchical Genetic Algorithm
    """
    def __init__(self, mst_edges, num_subtree=4, subtree_size_min=4, group_size=50, num_iter=4, iter_size=100):
        """
        Initialize the tree splitter with genetic algorithm parameters
        
        Parameters:
        mst_edges -- List of minimum spanning tree edges [(u,v,weight)]
        num_subtree -- Number of subtrees to split into (default 4)
        subtree_size_min -- Minimum number of nodes per subtree (default 4)
        group_size -- Population size for each group (default 50)
        num_iter -- Number of hierarchical iterations (default 4)
        iter_size -- Number of iterations per hierarchy level (default 100)
        """
        group_size = int(group_size)
        num_iter = int(num_iter)
        iter_size = int(iter_size)

        self.mst_edges = mst_edges
        self.num_subtree = num_subtree
        self.subtree_size_min = subtree_size_min
        self.group_size = group_size//2*2  # Ensure population size is even
        self.num_iter = num_iter
        self.iter_size = iter_size

        # Build graph
        self.G = nx.Graph()
        for u, v, w in mst_edges:
            self.G.add_edge(u, v, weight=w)

        self.edges = list(self.G.edges())
        self.num_edges = len(self.edges)
        self.num_nodes = len(self.G.nodes())

        # Validate parameters
        if self.num_nodes < num_subtree * subtree_size_min:
            raise ValueError(f"Cannot split: Need at least {num_subtree * subtree_size_min} nodes, but only have {self.num_nodes}")

        # Genetic algorithm parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_ratio = 0.2  # Elite retention ratio

    def get_deletable_edges(self) -> List[int]:
        """
        Get edges that can be deleted without causing any subtree to have fewer than subtree_size_min nodes
        
        Returns:
            List of deletable edge indices
        """
        global deletable_edges
        deletable_edges = []

        # Set up progress reporting
        total_edges = len(self.edges)
        arcpy.SetProgressor("step", "Identifying deletable edges...", 0, total_edges, 1)
        arcpy.AddMessage(f"Analyzing {total_edges} edges to find deletable ones...")

        for i, edge in enumerate(self.edges):
            # Temporarily remove edge
            temp_G = self.G.copy()
            temp_G.remove_edge(*edge)

            # Check connected components
            components = list(nx.connected_components(temp_G))

            # Edge is deletable only if all components have at least n nodes
            if all(len(c) >= self.subtree_size_min for c in components):
                deletable_edges.append(i)
            
            # Update progress
            arcpy.SetProgressorPosition(i + 1)
            if (i + 1) % max(1, total_edges // 10) == 0:  # Report every 10%
                arcpy.AddMessage(f"Processed {i + 1} of {total_edges} edges ({((i + 1) / total_edges) * 100:.1f}%)")

        arcpy.ResetProgressor()
        arcpy.AddMessage(f"Found {len(deletable_edges)} deletable edges out of {total_edges} total edges")
        return deletable_edges

    def run_ga(self) -> Tuple[List[List[Tuple[int, int, float]]], List[Tuple[int, int, float]], float]:
        """
        Main hierarchical genetic algorithm function
        
        Returns:
            Tuple containing:
            - List of subtrees (each subtree is a list of edges)
            - List of removed edges
            - Score (modularity value)
        """
        group_num = 2 ** (self.num_iter - 1)    # Number of groups based on iteration layers
        groups = []     # Store all group information
        
        arcpy.AddMessage(f"Starting genetic algorithm with {group_num} groups of {self.group_size} individuals each")
        arcpy.AddMessage("Initializing populations...")
        
        # Initialize all groups with random valid individuals
        for g in range(group_num):
            population = []
            attempts = 0
            max_attempts = self.group_size * 10  # Limit attempts to prevent infinite loops
            
            while len(population) < self.group_size and attempts < max_attempts:
                # Randomly select N-1 edges
                edge_indices = random.sample(deletable_edges, self.num_subtree - 1)
                fitness = self.evaluate_fitness(edge_indices)
                if fitness > -1:  # Only accept valid individuals
                    population.append(edge_indices)
                attempts += 1
            
            if len(population) < self.group_size:
                arcpy.AddWarning(f"Group {g+1}: Could only generate {len(population)} valid individuals (target: {self.group_size})")
            
            fitnesses = [self.evaluate_fitness(ind) for ind in population]
            groups.append({
                'population': population,
                'fitnesses': fitnesses,
                'best_individual': population[np.argmax(fitnesses)],
                'best_fitness': max(fitnesses)
            })
            
            arcpy.AddMessage(f"Group {g+1} initialized with {len(population)} individuals, best fitness: {max(fitnesses):.6f}")

        # Ensure at least one valid group
        if not groups or all(g['best_fitness'] == -float('inf') for g in groups):
            raise ValueError("No valid individuals in initial population, please check parameters or deletable edges")

        # Global best
        global_best_group = max(groups, key=lambda g: g['best_fitness'])
        global_best = global_best_group['best_individual']
        global_score = global_best_group['best_fitness']
        
        arcpy.AddMessage(f"Initial global best fitness: {global_score:.6f}")
        
        # Set up progress reporting for generations
        total_generations = self.num_iter * self.iter_size
        arcpy.SetProgressor("step", "Running genetic algorithm...", 0, total_generations, 1)

        for iteration in range(1, total_generations + 1):
            if iteration == total_generations:
                # Last generation, only one group left, no need to sort
                pass
            elif iteration % self.iter_size == 0:
                # End of each iteration level, keep half of the groups
                groups.sort(key=lambda g: g['best_fitness'])
                now_group_num = len(groups)
                groups = groups[int(now_group_num / 2):]
                arcpy.AddMessage(f"End of level {iteration // self.iter_size}: Keeping top {len(groups)} groups")

            # Update each group
            for group_idx, group in enumerate(groups):
                self._update_group(group, deletable_edges)

            # Update global best
            current_best_group = max(groups, key=lambda g: g['best_fitness'])
            if current_best_group['best_fitness'] > global_score:
                global_best = current_best_group['best_individual']
                global_score = current_best_group['best_fitness']
                arcpy.AddMessage(f"Generation {iteration}: New global best fitness: {global_score:.6f}")
            
            # Update progress
            arcpy.SetProgressorPosition(iteration)
            if iteration % max(1, total_generations // 20) == 0:  # Report every 5%
                arcpy.AddMessage(f"Generation {iteration}/{total_generations} complete ({(iteration / total_generations) * 100:.1f}%)")

        arcpy.ResetProgressor()
        arcpy.AddMessage(f"Genetic algorithm completed with final best fitness: {global_score:.6f}")

        # Build and return results
        subtrees, removed_edges = self._build_result(global_best)
        return subtrees, removed_edges, global_score

    def _update_group(self, group: Dict, deletable_edges: List[int]) -> None:
        """
        Update a single population group through selection, crossover and mutation
        
        Parameters:
            group -- Dictionary containing population and fitness information
            deletable_edges -- List of edge indices that can be deleted
        """
        population = group['population']    # All individuals in the population
        fitnesses = group['fitnesses']      # Fitness values for all individuals

        # Select elites - new population includes elites from previous generation
        elite_size = int(self.elite_ratio * len(population))
        elite_indices = np.argsort(fitnesses)[-elite_size:]  # Sort by fitness, select top individuals
        new_population = [population[i] for i in elite_indices]
        
        # Crossover and mutation to fill the rest of the population
        while len(new_population) < len(population):
            # Higher probability to select parents with higher fitness/modularity
            weights = np.array(fitnesses) / sum(fitnesses)
            parent1_index, parent2_index = np.random.choice(range(len(population)), p=weights, size=2, replace=False)
            parent1, parent2 = population[parent1_index], population[parent2_index]
            fitness1, fitness2 = fitnesses[parent1_index], fitnesses[parent2_index]
            
            # Crossover
            if len(parent1) <= 2:    # If individual has 2 or fewer genes, skip crossover
                child1, fit_child1, child2, fit_child2 = parent1, fitness1, parent2, fitness2
            else:
                child1, fit_child1, child2, fit_child2 = self.crossover(parent1, fitness1, parent2, fitness2)
                
            # Mutation
            child1, fit_child1 = self.mutate(child1, fit_child1, deletable_edges)
            child2, fit_child2 = self.mutate(child2, fit_child2, deletable_edges)

            new_population.extend([child1, child2])

        # Evaluate new population
        new_population = new_population[:len(population)]
        new_fitnesses = [self.evaluate_fitness(ind) for ind in new_population]

        # Update group information
        group['population'] = new_population
        group['fitnesses'] = new_fitnesses
        best_idx = np.argmax(new_fitnesses)
        group['best_individual'] = new_population[best_idx]
        group['best_fitness'] = new_fitnesses[best_idx]

    def evaluate_fitness(self, individual: List[int]) -> float:
        """
        Evaluate individual fitness (higher is better)
        
        Parameters:
            individual -- List of edge indices to remove
            
        Returns:
            Fitness value (modularity score, or negative value if invalid)
        """
        try:
            edge_indices = [int(x) for x in individual]
            temp_G = self.G.copy()
            for idx in edge_indices:
                edge = self.edges[idx]
                temp_G.remove_edge(*edge)
                
            components = list(nx.connected_components(temp_G))
            cc_list = [list(c) for c in components]
            Q = round(modularity(self.G, cc_list), 4)
            return max(Q, 0.00001)  # Ensure return value is a small positive number
        except:
            return -10

    def crossover(self, parent1: List[int], fitness1: float, 
                  parent2: List[int], fitness2: float) -> Tuple[List[int], float, List[int], float]:
        """
        Perform crossover between two parents
        
        Parameters:
            parent1, parent2 -- Parent individuals (lists of edge indices)
            fitness1, fitness2 -- Fitness values of parents
            
        Returns:
            Tuple of (child1, fitness1, child2, fitness2)
        """
        count = 0
        while count < 10:   # Try crossover 10 times, return parents if no better offspring found
            count += 1
            size = len(parent1)
            child1, child2 = set(), set()
            
            # Select crossover points
            start, end = sorted(random.sample(range(size), 2))

            # Copy crossover section from parents
            child1.update(parent1[start:end + 1])
            child2.update(parent2[start:end + 1])

            # Fill remaining positions
            def fill_child(child, parent):
                # Update child with parent data
                child.update(random.sample(list(set(parent)-child), k=size-len(child)))
                return list(child)

            child1 = fill_child(child1, parent2)
            child2 = fill_child(child2, parent1)

            fit_child1, fit_child2 = self.evaluate_fitness(child1), self.evaluate_fitness(child2)
            
            # Return best combinations
            if fit_child1 > fitness1 and fit_child2 > fitness2:
                return child1, fit_child1, child2, fit_child2
            elif fit_child1 > fitness1:
                return child1, fit_child1, parent2, fitness2
            elif fit_child2 > fitness2:
                return parent1, fitness1, child2, fit_child2
                
        return parent1, fitness1, parent2, fitness2

    def mutate(self, individual: List[int], fit_individual: float, 
               deletable_edges: List[int]) -> Tuple[List[int], float]:
        """
        Perform mutation on an individual
        
        Parameters:
            individual -- Individual to mutate
            fit_individual -- Fitness of the individual
            deletable_edges -- List of deletable edge indices
            
        Returns:
            Tuple of (mutated_individual, fitness)
        """
        if random.random() > self.mutation_rate:
            return individual, fit_individual
            
        count = 0
        while count < 10:
            count += 1
            mutated = individual.copy()
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = random.choice(deletable_edges)
            mutated_fitness = self.evaluate_fitness(mutated)
            if mutated_fitness > fit_individual:
                return mutated, mutated_fitness
                
        return individual, fit_individual

    def _build_result(self, edge_indices: List[int]) -> Tuple[List[List[Tuple[int, int, float]]], List[Tuple[int, int, float]]]:
        """
        Build final result from best individual
        
        Parameters:
            edge_indices -- List of edge indices to remove
            
        Returns:
            Tuple containing:
            - List of subtrees (each subtree is a list of edges)
            - List of removed edges
        """
        temp_G = self.G.copy()
        removed_edges = []
        
        for idx in edge_indices:
            edge = self.edges[idx]
            temp_G.remove_edge(*edge)
            removed_edges.append((edge[0], edge[1], self.G[edge[0]][edge[1]]['weight']))

        components = list(nx.connected_components(temp_G))
        subtrees = []

        for comp in components:
            subtree = []
            for u, v, w in self.mst_edges:
                if u in comp and v in comp:
                    subtree.append((u, v, w))

            # Handle isolated nodes
            if not subtree:
                node = next(iter(comp))  # Get the single node
                subtree.append((node, node, -1))  # Mark isolated node

            subtrees.append(subtree)

        return subtrees, removed_edges


def read_neighbors_file(file_path: str) -> Tuple[np.ndarray, Set[int], int]:
    """
    Read neighbor file and generate adjacency matrix
    
    Parameters:
        file_path (str): Path to neighbor file
        
    Returns:
        tuple: (adjacency matrix, set of nodes, matrix size)
    """
    arcpy.AddMessage(f"Reading neighbor information from {file_path}...")
    
    with open(file_path, 'r') as f:
        # Initialize adjacency dictionary
        neighbors_dict = {}
        # Maximum node ID
        max_node = 0
        # Set of all nodes
        nodes_set = set()
        # Read all lines and process
        lines_processed = 0
        
        for line in f:
            nodes = list(map(int, line.strip().split(',')))
            current_node = nodes[0]
            neighbors = nodes[1:]
            # Update maximum node ID
            max_node = max(max(nodes), max_node)
            # Count all nodes
            nodes_set.add(current_node)
            # Store neighbor relationships
            neighbors_dict[current_node] = neighbors
            lines_processed += 1
            
        arcpy.AddMessage(f"Processed {lines_processed} neighbor records")

        # Determine total number of nodes (assume node IDs start at 0 and are continuous)
        matrix_len = max_node + 1
        # Initialize adjacency matrix (all zeros)
        adjacency_matrix = np.zeros((matrix_len, matrix_len), dtype=int)

        # Fill adjacency matrix
        arcpy.AddMessage("Building adjacency matrix...")
        for node in neighbors_dict:
            for neighbor in neighbors_dict[node]:
                adjacency_matrix[node][neighbor] = 1
                adjacency_matrix[neighbor][node] = 1  # Undirected graph needs to be symmetric

        arcpy.AddMessage(f"Created adjacency matrix of size {matrix_len}x{matrix_len} with {len(nodes_set)} active nodes")
        return adjacency_matrix, nodes_set, matrix_len


def read_weight_file(file_path: str, nodes_size: Set[int], matrix_len: int) -> np.ndarray:
    """
    Read directed graph weight file and generate undirected weight matrix (summing bidirectional weights)
    
    Parameters:
        file_path (str): Path to weight file
        nodes_size (Set[int]): Set of valid node IDs
        matrix_len (int): Size of the matrix to create
        
    Returns:
        np.ndarray: Undirected weight matrix where weight between node1-node2 = node1->node2 + node2->node1
                   Unconnected nodes have weight 0
    """
    arcpy.AddMessage(f"Reading weight information from {file_path}...")
    
    with open(file_path, 'r') as f:
        # Skip header row
        next(f)

        # Use dictionary to store all edge relationships
        edge_dict = {}
        node_set = set()
        lines_processed = 0

        # Read and process all lines
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # Skip malformatted lines

            node1 = int(parts[0])
            node2 = int(parts[1])
            weight = float(parts[2])

            node_set.add(node1)
            node_set.add(node2)

            # Store edge relationship (considering direction)
            edge_dict[(node1, node2)] = weight
            lines_processed += 1

        arcpy.AddMessage(f"Processed {lines_processed} weight records")

        # Initialize weight matrix (all zeros)
        weight_matrix = np.zeros((matrix_len, matrix_len))

        # Fill weight matrix (consider bidirectional weight sum)
        arcpy.AddMessage("Building weight matrix...")
        edges_processed = 0
        
        for i in nodes_size:
            for j in nodes_size:  # Only process upper triangle to avoid duplicate calculation
                if i == j:
                    continue
                # Get weights for both directions
                weight_ij = edge_dict.get((i, j), 0)
                weight_ji = edge_dict.get((j, i), 0)

                # Undirected graph weight is sum of both directions
                total_weight = weight_ij + weight_ji

                if total_weight > 0:
                    weight_matrix[i][j] = total_weight
                    weight_matrix[j][i] = total_weight  # Symmetric assignment
                    edges_processed += 1

        arcpy.AddMessage(f"Created weight matrix with {edges_processed} weighted edges")
        return weight_matrix


def prim_mst(adj_matrix: np.ndarray, weight_matrix: np.ndarray, nodes_set: Set[int]) -> List[Tuple[int, int, float]]:
    """
    Prim's algorithm to generate maximum spanning tree (handles disconnected graphs)
    
    Parameters:
        adj_matrix -- Adjacency matrix (0/1 matrix)
        weight_matrix -- Weight matrix
        nodes_set -- Set of nodes
        
    Returns:
        largest_mst -- List of edges in the largest maximum spanning tree (as tuples: (i,j,weight))
    """
    arcpy.AddMessage("Generating maximum spanning tree using Prim's algorithm...")
    
    n = adj_matrix.shape[0]
    all_msts = []  # Store spanning trees for all connected components
    unvisited = set(nodes_set)  # Set of unvisited nodes
    
    component_count = 0
    
    while unvisited:
        component_count += 1
        # Randomly select an unvisited node as starting point
        start_node = random.choice(list(unvisited))
        visited = {start_node}
        mst_edges = []
        candidate_edges = []  # Candidate edge heap (simulated with list)

        # Initialize candidate edges (all edges from starting node)
        for j in range(n):
            if adj_matrix[start_node][j] == 1:
                weight = float(weight_matrix[start_node][j])
                candidate_edges.append((-weight, start_node, j))  # Use negative values to simulate max heap

        # Main Prim's algorithm loop
        while candidate_edges:
            # Get edge with current maximum weight
            candidate_edges.sort()  # Sort by weight in ascending order (negative values)
            neg_weight, i, j = candidate_edges.pop(0)
            weight = -neg_weight

            if j not in visited:
                mst_edges.append((i, j, weight))
                visited.add(j)

                # Add new node's edges to candidates
                for k in range(n):
                    if adj_matrix[j][k] == 1 and k not in visited:
                        new_weight = float(weight_matrix[j][k])
                        candidate_edges.append((-new_weight, j, k))

        # Record current component's spanning tree
        all_msts.append(mst_edges)
        unvisited -= visited
        
        arcpy.AddMessage(f"Component {component_count}: {len(visited)} nodes, {len(mst_edges)} edges")

    # Find spanning tree with most nodes
    largest_mst = max(all_msts, key=lambda mst: len(set().union(*[(u, v) for u, v, _ in mst])))
    
    arcpy.AddMessage(f"Found maximum spanning tree with {len(largest_mst)} edges")
    return largest_mst


def apply_unique_value_renderer_complete(output_fc: str, field_name: str = "subregion_id", 
                                       color_ramp_name: str = "Prismatic") -> None:
    """
    Apply unique value renderer to polygon feature class
    
    Parameters:
    - output_fc: Feature class path
    - field_name: Rendering field name
    - color_ramp_name: Color scheme name
    """
    arcpy.AddMessage(f"Applying unique value renderer to {output_fc} using field {field_name}...")
    
    # Get current project and map
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    map_obj = aprx.activeMap
    
    # Check if layer already exists
    layer_name = os.path.basename(output_fc)
    existing_layers = [lyr.name for lyr in map_obj.listLayers()]
    
    if layer_name in existing_layers:
        # If exists, get the layer
        layer = map_obj.listLayers(layer_name)[0]
        arcpy.AddMessage(f"Using existing layer: {layer_name}")
    else:
        # Add feature class to map
        layer = map_obj.addDataFromPath(output_fc)
        arcpy.AddMessage(f"Added new layer: {layer_name}")
    
    # Get symbology system
    sym = layer.symbology
    
    # Check if renderer is supported
    if hasattr(sym, 'renderer'):
        # Update to unique value renderer
        sym.updateRenderer('UniqueValueRenderer')
        renderer = sym.renderer
        
        # Set rendering field
        renderer.fields = [field_name]
        
        # Get available color schemes
        aprx_path = aprx.filePath
        styles = arcpy.mp.ArcGISProject(aprx_path).listColorRamps(color_ramp_name)
        
        if styles:
            # Apply color scheme
            renderer.colorRamp = styles[0]
            arcpy.AddMessage(f"Applied color ramp: {color_ramp_name}")
        
        # Update symbology
        layer.symbology = sym
        
        arcpy.AddMessage(f"Successfully applied unique value renderer")
        arcpy.AddMessage(f"Field: {field_name}")
        arcpy.AddMessage(f"Number of unique values: {len(renderer.groups[0].items)}")
        
    else:
        arcpy.AddMessage("This layer does not support renderer operations")
    
    return layer


# Calculate Euclidean distance
def compute_edis_distance(ts1: List[Tuple[float, str]], ts2: List[Tuple[float, str]]) -> float:
    """
    Calculate Euclidean distance between two time series
    
    Parameters:
        ts1, ts2: Two time series as lists of (value, timestamp) tuples
        
    Returns:
        Dissimilarity score (higher means more different)
    """
    # Extract value sequences
    values1 = [v for v, _ in ts1]
    values2 = [v for v, _ in ts2]

    # Determine length of shorter sequence
    min_length = min(len(values1), len(values2))

    # Calculate similarity for matching portion
    matching_similarity = sum(abs(values1[i] - values2[i]) for i in range(min_length))

    # Calculate average similarity
    average_similarity = matching_similarity / min_length

    # Calculate length of unmatched portion
    unmatched_length = abs(len(values1) - len(values2))

    # Calculate similarity for unmatched portion
    unmatched_similarity = average_similarity * unmatched_length

    # Calculate total similarity
    total_similarity = matching_similarity + unmatched_similarity

    return total_similarity


# Calculate cosine similarity
def compute_cosine_similarity(ts1: List[Tuple[float, str]], ts2: List[Tuple[float, str]]) -> float:
    """
    Calculate cosine similarity between two time series
    
    Parameters:
        ts1, ts2: Two time series as lists of (value, timestamp) tuples
        
    Returns:
        Dissimilarity score (higher means more different)
    """
    # Extract value sequences
    values1 = [v for v, _ in ts1]
    values2 = [v for v, _ in ts2]

    # Determine length of shorter sequence
    min_length = min(len(values1), len(values2))

    # Calculate dot product
    dot_product = sum(values1[i] * values2[i] for i in range(min_length))
    
    # Calculate magnitudes of both vectors
    magnitude1 = sum(v**2 for v in values1[:min_length])**0.5
    magnitude2 = sum(v**2 for v in values2[:min_length])**0.5
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Calculate cosine similarity for matching portion (1 = completely similar, 0 = orthogonal, -1 = completely opposite)
    matching_similarity = dot_product / (magnitude1 * magnitude2)
    
    # Handle unmatched portion (unmatched portion reduces overall similarity)
    unmatched_length = abs(len(values1) - len(values2))
    
    # Adjust similarity based on unmatched length (more unmatched, lower similarity)
    if unmatched_length > 0 and (len(values1) + len(values2)) > 0:
        adjustment_factor = min_length / (min_length + unmatched_length)
        total_similarity = matching_similarity * adjustment_factor
    else:
        total_similarity = matching_similarity
    
    # Return dissimilarity (1 - similarity)
    return 1 - total_similarity


# DTW measure of difference between sequences, extract only values from time series, ignore timestamps
def compute_dtw_distance(ts1: List[Tuple[float, str]], ts2: List[Tuple[float, str]]) -> float:
    """
    Calculate Dynamic Time Warping distance between two time series
    
    Parameters:
        ts1, ts2: Two time series as lists of (value, timestamp) tuples
        
    Returns:
        DTW distance (higher means more different)
    """
    # Extract value sequences
    seq1 = [item[0] for item in ts1]
    seq2 = [item[0] for item in ts2]

    n, m = len(seq1), len(seq2)

    # Initialize cost matrix
    # First row and column set to infinity except for position (0,0)
    cost_matrix = np.zeros((n + 1, m + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            try:
                # Calculate distance between current points
                cost = abs(seq1[i - 1] - seq2[j - 1])
            except Exception as e:
                arcpy.AddError(f"Error in DTW calculation: {e}")
                return float('inf')
                
            # Choose minimum cost path
            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j],    # Insertion
                                          cost_matrix[i, j - 1],     # Deletion
                                          cost_matrix[i - 1, j - 1]) # Match

    # Return value in bottom-right corner, the total DTW distance
    return cost_matrix[n, m]


# Calculate periodicity difference between two irregularly sampled time series
def compute_period_distance(ts1: List[Tuple[float, str]], ts2: List[Tuple[float, str]], 
                           fmin: float = 0.1, fmax: float = 10, n_freqs: int = 1000) -> float:
    """
    Calculate periodicity similarity between two irregularly sampled time series
    
    Parameters:
        ts1, ts2: Two time series as lists of (value, timestamp) tuples
        fmin, fmax: Minimum and maximum frequency range
        n_freqs: Number of frequency points to calculate
        
    Returns:
        Dissimilarity score (higher means more different)
    """
    def preprocess_timeseries(ts):
        """
        Preprocess time series data
        
        Parameters:
            ts: Input time series in format [(value, timestamp), ...]
            
        Returns:
            values: Array of values
            timestamps: Array of corresponding datetime objects
        """
        values, timestamps = zip(*ts)
        values = np.array(values)
        timestamps = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in timestamps])
        return values, timestamps

    # Preprocess time series
    try:
        values1, timestamps1 = preprocess_timeseries(ts1)
        values2, timestamps2 = preprocess_timeseries(ts2)

        # Convert timestamps to relative seconds
        timestamps1_seconds = np.array([(t - timestamps1[0]).total_seconds() for t in timestamps1])
        timestamps2_seconds = np.array([(t - timestamps2[0]).total_seconds() for t in timestamps2])

        # Calculate time series characteristics
        duration1 = timestamps1_seconds[-1] - timestamps1_seconds[0]
        duration2 = timestamps2_seconds[-1] - timestamps2_seconds[0]

        min_interval1 = np.min(np.diff(timestamps1_seconds))
        min_interval2 = np.min(np.diff(timestamps2_seconds))

        # Adjust frequency range based on time series characteristics
        fmin = max(fmin, 1 / max(duration1, duration2))
        fmax = min(fmax, 1 / (2 * min(min_interval1, min_interval2)))

        # Create frequency array
        frequencies = np.linspace(fmin, fmax, n_freqs)

        # Calculate Lomb-Scargle periodogram
        power1 = signal.lombscargle(timestamps1_seconds, values1, frequencies)
        power2 = signal.lombscargle(timestamps2_seconds, values2, frequencies)

        # Normalize power spectra
        power1_norm = power1 / np.sum(power1)
        power2_norm = power2 / np.sum(power2)

        # Calculate cosine similarity as periodicity similarity score
        similarity = np.dot(power1_norm, power2_norm) / (np.linalg.norm(power1_norm) * np.linalg.norm(power2_norm))

        # Convert similarity to dissimilarity
        dissimilarity = 1 - similarity
        
        return dissimilarity
    except Exception as e:
        arcpy.AddWarning(f"Error in period distance calculation: {e}")
        return 1.0  # Return maximum dissimilarity on error


# Measure shape difference
def compute_shape_distance(TS_1: List[Tuple[float, str]], TS_2: List[Tuple[float, str]]) -> float:
    """
    Calculate shape distance between two time series
    
    Parameters:
        TS_1, TS_2: Two time series as lists of (value, timestamp) tuples
        
    Returns:
        Dissimilarity score (higher means more different)
    """
    def parse_datetime(date_string):
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    # Equalize two sequences
    def equalize_time_series(ts1, ts2):
        ts1 = [(v, parse_datetime(t)) for v, t in ts1]
        ts2 = [(v, parse_datetime(t)) for v, t in ts2]

        def find_time_intersection(ts1, ts2):
            start1, end1 = ts1[0][1], ts1[-1][1]
            start2, end2 = ts2[0][1], ts2[-1][1]
            intersection_start = max(start1, start2)
            intersection_end = min(end1, end2)
            return intersection_start, intersection_end

        def interpolate_value(t, t1, t2, v1, v2):
            if t1 == t2:
                return v1
            ratio = ((t - t1).total_seconds() / 60) / ((t2 - t1).total_seconds() / 60)
            return v1 + ratio * (v2 - v1)

        intersection_start, intersection_end = find_time_intersection(ts1, ts2)
        all_timestamps = sorted(set([t for _, t in ts1 + ts2 if intersection_start <= t <= intersection_end]))

        def interpolate_series(ts):
            result = []
            for t in all_timestamps:
                if t < ts[0][1] or t > ts[-1][1]:
                    continue
                i = bisect.bisect_left([x[1] for x in ts], t)
                if i == 0 or ts[i][1] == t:
                    result.append((ts[i][0], t))
                else:
                    v = interpolate_value(t, ts[i - 1][1], ts[i][1], ts[i - 1][0], ts[i][0])
                    result.append((v, t))
            return result

        ts1_eq = interpolate_series(ts1)
        ts2_eq = interpolate_series(ts2)

        min_len = min(len(ts1_eq), len(ts2_eq))
        ts1_eq = ts1_eq[:min_len]
        ts2_eq = ts2_eq[:min_len]

        ts1_eq = [(v, t.strftime("%Y-%m-%d %H:%M:%S")) for v, t in ts1_eq]
        ts2_eq = [(v, t.strftime("%Y-%m-%d %H:%M:%S")) for v, t in ts2_eq]

        return ts1_eq, ts2_eq

    def calculate_similarity(ts1, ts2):
        patterns1, time_lengths1, _ = pattern_analysis(ts1)
        patterns2, time_lengths2, _ = pattern_analysis(ts2)

        min_length = min(len(patterns1), len(patterns2))
        patterns1 = patterns1[:min_length]
        patterns2 = patterns2[:min_length]
        time_lengths1 = time_lengths1[:min_length]
        time_lengths2 = time_lengths2[:min_length]

        total_time = sum(time_lengths1)
        dis_similarity = 0

        for i in range(min_length):
            t_wi = time_lengths1[i] / total_time
            pattern_diff = abs(patterns1[i] - patterns2[i])
            value_diff = abs((ts1[i + 2][0] - ts1[i + 1][0]) - (ts2[i + 2][0] - ts2[i + 1][0]))
            dis_similarity += t_wi * pattern_diff * value_diff
        return dis_similarity

    def pattern_analysis(time_series, epsilon=math.tan(math.pi / 360)):
        def calculate_slope(t1, t2, v1, v2):
            time_diff = (parse_datetime(t2) - parse_datetime(t1)).total_seconds() / 60
            return (v2 - v1) / time_diff if time_diff != 0 else 0

        def determine_pattern(k_current, k_next, epsilon):
            if (k_next > epsilon and k_current < epsilon) or (
                    k_next > epsilon and k_current > epsilon and k_next - k_current > 0):
                return 3  # Accelerating increase
            elif k_next > epsilon and k_current > epsilon and k_next - k_current == 0:
                return 2  # Constant increase
            elif k_next > epsilon and k_current > epsilon and k_next - k_current < 0:
                return 1  # Decelerating increase
            elif (k_next < -epsilon and k_current > -epsilon) or (
                    k_next < -epsilon and k_current < -epsilon and k_next - k_current < 0):
                return -3  # Accelerating decrease
            elif k_next < -epsilon and k_current < -epsilon and k_next - k_current == 0:
                return -2  # Constant decrease
            elif k_next < -epsilon and k_current < -epsilon and k_next - k_current > 0:
                return -1  # Decelerating decrease
            elif -epsilon < k_next < epsilon:
                return 0  # No change
            else:
                return None  # Undefined pattern

        patterns = []
        time_lengths = []
        slopes = []

        for i in range(len(time_series) - 2):
            v1, t1 = time_series[i]
            v2, t2 = time_series[i + 1]
            v3, t3 = time_series[i + 2]

            k_current = calculate_slope(t1, t2, v1, v2)
            k_next = calculate_slope(t2, t3, v2, v3)

            pattern = determine_pattern(k_current, k_next, epsilon)
            if pattern is not None:
                patterns.append(pattern)
                time_lengths.append((parse_datetime(t2) - parse_datetime(t1)).total_seconds() / 60)
                slopes.append(k_current)

        return patterns, time_lengths, slopes

    try:
        # Equalize sequences
        TS_1_eq, TS_2_eq = equalize_time_series(TS_1, TS_2)

        # Pattern analysis
        patterns1, _, slopes1 = pattern_analysis(TS_1_eq)
        patterns2, _, slopes2 = pattern_analysis(TS_2_eq)

        # Calculate similarity
        dis_similarity = calculate_similarity(TS_1_eq, TS_2_eq)
        return dis_similarity
    except Exception as e:
        arcpy.AddWarning(f"Error in shape distance calculation: {e}")
        return 1.0  # Return maximum dissimilarity on error


def create_lengthwise_time_series_list_by_specific_column(file_path: str, id_col: int = 0, 
                                                       value_col: int = 1, date_col: int = -1, 
                                                       scale_rate: float = 1) -> Dict[int, List[Tuple[float, str]]]:
    """
    Read time series information from given file path and create time series list
    
    Parameters:
        file_path (str): Path to file containing time series information
        id_col (int): Column containing ID
        value_col (int): Column containing values
        date_col (int): Column containing dates (-1 for auto-generation)
        scale_rate (float): Scaling ratio
        
    Returns:
        Dict[int, List[Tuple[float, str]]]: Dictionary of time series, keys are sequence IDs, 
                                          values are lists of (value, time_string) tuples
    """
    arcpy.AddMessage(f"Reading time series data from {file_path}...")

    # Compress time series based on given compression ratio
    def compress_time_series(ts: List[Tuple[float, str]], scale_rate: float) -> List[Tuple[float, str]]:
        """
        Compress time series based on given compression ratio
        
        Parameters:
            ts: Original time series as list of (value, timestamp) tuples
            scale_rate: Compression ratio (0 < scale_rate <= 1)
            
        Returns:
            Compressed time series
        """
        # Interpolate value based on surrounding values
        def interpolate_value(prev: float, current: float, next: float) -> float:
            """
            Interpolate new value based on surrounding values
            
            Parameters:
                prev: Previous value
                current: Current value
                next: Next value
                
            Returns:
                Interpolated new value
            """
            return round(float((prev + current + next) / 3), 3)

        # Interpolate timestamp based on surrounding timestamps
        def interpolate_timestamp(prev: str, current: str, next: str) -> str:
            """
            Interpolate new timestamp based on surrounding timestamps
            
            Parameters:
                prev: Previous timestamp
                current: Current timestamp
                next: Next timestamp
                
            Returns:
                Interpolated new timestamp
            """
            # Convert string timestamps to datetime objects
            prev_dt = datetime.strptime(prev, "%Y-%m-%d %H:%M:%S")
            current_dt = datetime.strptime(current, "%Y-%m-%d %H:%M:%S")
            next_dt = datetime.strptime(next, "%Y-%m-%d %H:%M:%S")

            # Calculate interpolated timestamp
            interpolated_dt = prev_dt + (next_dt - prev_dt) / 2
            return interpolated_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Check if compression ratio is valid
        if not 0 < scale_rate <= 1:
            raise ValueError("Compression ratio scale_rate must be between 0 and 1")

        # If time series length is 2 or less, no need to compress, return directly
        if len(ts) <= 2:
            return ts

        # Calculate number of elements to keep
        n = len(ts)
        keep_count = max(2, int(n * scale_rate))

        # Always keep first element
        result = [ts[0]]

        if keep_count > 2:
            # Calculate step for selecting elements
            step = (n - 2) / (keep_count - 2)

            # Interpolate middle elements
            for i in range(1, keep_count - 1):
                index = int(i * step)
                prev_index = int((i - 1) * step)
                next_index = min(int((i + 1) * step), n - 1)
                
                # Interpolate new value
                value = interpolate_value(ts[prev_index][0], ts[index][0], ts[next_index][0])

                # Interpolate new timestamp
                timestamp = interpolate_timestamp(ts[prev_index][1], ts[index][1], ts[next_index][1])

                result.append((value, timestamp))

        # Add last element
        result.append(ts[-1])
        return result

    # Initialize dictionary to store all time series
    series_dict = {}

    # Set start time (for two-column data)
    start_time = datetime(2024, 1, 1, 0, 0)

    try:
        # Try to open and read file
        with open(file_path, 'r') as file:
            # Read file content line by line
            lines_processed = 0
            series_count = 0
            
            for line in file:
                if not line.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    continue
                    
                # Remove whitespace and split by comma
                items = line.strip().split(',')

                # Skip if line has incorrect format
                if len(items) < 2 or len(items) > 3:
                    arcpy.AddWarning(f"Warning: Skipping malformatted line in time series file: {line.strip()}")
                    continue

                # Extract sequence number and value
                sequence_number = int(items[id_col])
                value = float(items[value_col])  # Convert value to decimal

                # Handle time information
                if date_col != -1:
                    # If three columns, use provided time
                    time_str = items[date_col]
                else:
                    # If two columns, generate time
                    if sequence_number not in series_dict:
                        # If new sequence, use start time
                        current_time = start_time
                    else:
                        # If sequence exists, add 5 minutes to last time
                        last_time = datetime.strptime(series_dict[sequence_number][-1][1], "%Y-%m-%d %H:%M:%S")
                        current_time = last_time + timedelta(minutes=5)

                    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

                # Add (value, time) tuple to corresponding sequence
                if sequence_number in series_dict:
                    series_dict[sequence_number].append((value, time_str))
                else:
                    series_dict[sequence_number] = [(value, time_str)]
                    series_count += 1
                
                lines_processed += 1
                
                # Progress reporting
                if lines_processed % 10000 == 0:
                    arcpy.AddMessage(f"Processed {lines_processed} lines...")

        arcpy.AddMessage(f"Processed {lines_processed} lines, found {series_count} unique time series")

    except FileNotFoundError:
        # If file doesn't exist, print error
        arcpy.AddError(f"Error: File '{file_path}' not found.")
        return {}
    except IOError:
        # If file can't be read (e.g., permission issues), print error
        arcpy.AddError(f"Error: Cannot read file '{file_path}'.")
        return {}
    except ValueError as e:
        # If error occurs during conversion (e.g., non-integer value), print error
        arcpy.AddError(f"Error: Problem processing data - {str(e)}")
        return {}

    # Compress time series based on given compression ratio
    if scale_rate < 1.0:
        arcpy.AddMessage(f"Compressing time series with scale rate {scale_rate}...")
        
    scaled_series_dict = {}
    for index, time_series in series_dict.items():
        scaled_time_series = compress_time_series(time_series, scale_rate)
        scaled_series_dict[index] = scaled_time_series

    arcpy.AddMessage(f"Created {len(scaled_series_dict)} time series")
    return scaled_series_dict


# Create weight matrix
def create_weight_matrix(scaled_series_dict: Dict[int, List[Tuple[float, str]]], weight_type: str = 'edis') -> np.ndarray:
    """
    Create weight matrix based on time series similarity
    
    Parameters:
        scaled_series_dict: Dictionary of time series
        weight_type: Similarity method ('edis', 'cosine', 'period', 'dtw', 'shape')
        
    Returns:
        Similarity matrix (higher values mean more similar)
    """
    weight_type_dict = {
        'edis': "compute_edis_distance", 
        'cosine': "compute_cosine_similarity", 
        'period': 'compute_period_distance', 
        'dtw': 'compute_dtw_distance', 
        'shape': 'compute_shape_distance'
    }
    
    key_list = list(scaled_series_dict.keys())
    key_max_value = max(key_list) + 1
    dis_matrix = np.zeros((key_max_value, key_max_value)) - 1
    unitArea_count = len(key_list)
    
    arcpy.AddMessage(f"Creating weight matrix using {weight_type} method for {unitArea_count} areas...")
    arcpy.SetProgressor("step", "Calculating similarities...", 0, unitArea_count, 1)
    
    for idx, i in enumerate(key_list):
        if (idx % max(1, unitArea_count // 20)) == 0:  # Report every 5%
            arcpy.SetProgressorPosition(idx)
            arcpy.AddMessage(f"Processing area {idx+1} of {unitArea_count} ({((idx+1) / unitArea_count) * 100:.1f}%)...")
            
        for j in key_list:
            if i != j and dis_matrix[i][j] == -1:
                dis_matrix[i][j] = dis_matrix[j][i] = eval(weight_type_dict[weight_type])(scaled_series_dict[i], scaled_series_dict[j])
    
    arcpy.ResetProgressor()
    
    # Convert distance to similarity
    max_dis = np.max(dis_matrix)
    dis_matrix[dis_matrix == -1] = max_dis
    sim_matrix = max_dis - dis_matrix
    
    arcpy.AddMessage(f"Weight matrix created with maximum similarity value: {np.max(sim_matrix):.4f}")
    return sim_matrix


# Write symmetric weight matrix to file
def write_weight_to_file(weight_matrix: np.ndarray, file: str) -> None:
    """
    Write symmetric weight matrix to file
    
    Parameters:
        weight_matrix: numpy.ndarray - Symmetric weight matrix
        file: str - Output file path
    """
    arcpy.AddMessage(f"Writing weight matrix to {file}...")

    # Check if matrix is symmetric
    if not np.allclose(weight_matrix, weight_matrix.T):
        arcpy.AddWarning("Warning: Weight matrix is not symmetric")

    with open(file, 'w') as f:
        # Write header
        f.write("n1,n2,weight\n")

        # Traverse upper triangular matrix (i < j)
        edge_count = 0
        for i in range(weight_matrix.shape[0]):
            for j in range(i + 1, weight_matrix.shape[1]):
                weight = round(weight_matrix[i, j], 4)   # Keep 4 decimal places
                if weight != 0:  # Only write non-zero weights
                    f.write(f"{i},{j},{weight}\n")
                    edge_count += 1

    arcpy.AddMessage(f"Successfully wrote {edge_count} weighted edges to file")


def create_neighbor_matrix_using_tools(input_fc: str, node_id: str) -> str:
    """
    Generate neighborhood relationships using spatial weights matrix tools
    
    Parameters:
        input_fc: Input feature class
        node_id: Node ID field
        
    Returns:
        Path to output neighbor matrix CSV file
    """
    arcpy.AddMessage("Generating neighborhood relationships using spatial weights matrix tools...")
    
    # Get default workspace and output directory
    default_ws = arcpy.env.workspace
    pre_dir = os.path.dirname(default_ws)
    
    # Set temporary files and final output paths
    temp_swm_file = os.path.join(pre_dir, "temp_weights.swm")
    temp_table = os.path.join(pre_dir, "temp_neighbor_table")
    neighbor_matrix_csv = os.path.join(pre_dir, "output_neighbor_metrix_fileaa.csv")
    
    # Step 1: Generate spatial weights matrix (.swm file)
    arcpy.AddMessage("Generating spatial weights matrix...")

    arcpy.stats.GenerateSpatialWeightsMatrix(
        Input_Feature_Class=input_fc,
        Unique_ID_Field=node_id,
        Output_Spatial_Weights_Matrix_File=temp_swm_file,
        Conceptualization_of_Spatial_Relationships="CONTIGUITY_EDGES_CORNERS",
        Distance_Method="EUCLIDEAN",
        Exponent=1,
        Threshold_Distance=None,
        Number_of_Neighbors=0,
        Row_Standardization="NO_STANDARDIZATION"
    )

    arcpy.AddMessage("Spatial weights matrix generated successfully!")
    
    # Step 2: Convert spatial weights matrix to table
    arcpy.AddMessage("Converting spatial weights matrix to table...")

    arcpy.stats.ConvertSpatialWeightsMatrixtoTable(
        Input_Spatial_Weights_Matrix_File=temp_swm_file,
        Output_Table=temp_table
    )

    arcpy.AddMessage("Spatial weights matrix converted to table successfully")
    
    # Step 3: Extract neighborhood relationships from table and save as CSV
    arcpy.AddMessage("Converting table data to CSV format...")
    neighbors = {}
    
    # Read neighborhood relationships from table
    with arcpy.da.SearchCursor(temp_table, [node_id, "NID"]) as cursor:
        for row in cursor:
            source_id = row[0]
            neighbor_id = row[1]
            
            if source_id not in neighbors:
                neighbors[source_id] = set()
                
            neighbors[source_id].add(neighbor_id)
    
    # Write results to CSV file
    with open(neighbor_matrix_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for region_id, neighbor_ids in neighbors.items():
            row = [region_id] + list(neighbor_ids)
            writer.writerow(row)
    
    arcpy.AddMessage(f"Neighborhood relationships saved to: {neighbor_matrix_csv}")
    arcpy.AddMessage(f"Found {len(neighbors)} areas with neighborhood relationships")
    
    # Clean up temporary files
    if arcpy.Exists(temp_table):
        arcpy.Delete_management(temp_table)
        arcpy.AddMessage("Temporary table deleted")
        
    if os.path.exists(temp_swm_file):
        os.remove(temp_swm_file)
        arcpy.AddMessage("Temporary SWM file deleted")
    
    return neighbor_matrix_csv


def script_tool(input_areal_featurelayer: str, area_id_field: str, input_sts_table: str, 
               area_id: str, value: str, time_series: str, seq: str, num_subtree: str, 
               subtree_size_min: str, similarity_method: str, out_feature_class: str, 
               group_size: str, num_iter: str, iter_size: str) -> None:
    """
    Main script tool function
    
    Parameters:
        input_areal_featurelayer: Input areal feature layer
        area_id_field: Area ID field
        input_sts_table: Input space-time series table
        area_id: Area ID field in time series table
        value: Value field in time series table
        time_series: Time series field in time series table
        seq: Sequence field
        num_subtree: Number of subtrees to create
        subtree_size_min: Minimum size of each subtree
        similarity_method: Method for calculating similarity
        out_feature_class: Output feature class
        group_size: Population size for genetic algorithm
        num_iter: Number of iterations for genetic algorithm
        iter_size: Size of each iteration
    """
    # Import needed at runtime to avoid startup issues
    import random
    arcpy.AddMessage("Starting space-time series regionalization tool...")
    start_time_total = time.time()

    # Create temporary files
    arcpy.AddMessage("Traversing each unit to find neighboring units...")
    start_time = time.time()
    neighbor_matrix_csv = create_neighbor_matrix_using_tools(input_areal_featurelayer, area_id_field)
    end_time = time.time()
    arcpy.AddMessage(f"Neighborhood traversal completed in {end_time - start_time:.2f} seconds!")
    arcpy.AddMessage("----------------------------------")

    # Get default workspace and output directory
    default_ws = arcpy.env.workspace
    pre_dir = os.path.dirname(default_ws)

    arcpy.AddMessage("Processing space-time series file...")
    start_time = time.time()
    
    # Create space-time series data file
    sts_csv = os.path.join(pre_dir, "output_sts_file.csv")

    # Open CSV file for writing
    with open(sts_csv, 'w', newline='') as csvfile:
        # Create CSV writer
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(["area_id", "value", "time_series"])

        # Use arcpy SearchCursor to read data from input table
        i = 0
        with arcpy.da.SearchCursor(input_sts_table, [area_id, value, time_series]) as cursor:
            # Traverse each row and write to CSV
            for row in cursor:
                # Create a new row list
                new_row = list(row)
                
                # Check and format time_series field (assume it's the 3rd column, index 2)
                if isinstance(new_row[2], datetime):
                    # Format datetime to string without decimal point
                    new_row[2] = new_row[2].strftime("%Y-%m-%d %H:%M:%S")
                
                # Write formatted row
                csv_writer.writerow(new_row)
                i = i + 1

    # Create weight matrix data file
    sts_weight_matrix_csv = os.path.join(pre_dir, "sts_weight_matrix_file.csv")

    # Read data file
    file_path = sts_csv
    scaled_series_dict = create_lengthwise_time_series_list_by_specific_column(file_path, id_col=0, value_col=1, date_col=2, scale_rate=1)

    weight_matrix = np.array([])

    # Create weight matrix based on data file, higher weight means more similar
    arcpy.AddMessage(f"Creating similarity matrix using {similarity_method} method...")
    if similarity_method == 'compute_edis_distance':
        weight_matrix = create_weight_matrix(scaled_series_dict, 'edis')
    elif similarity_method == "compute_cosine_distance":
        weight_matrix = create_weight_matrix(scaled_series_dict, 'cosine')
    elif similarity_method == 'compute_dtw_distance':
        weight_matrix = create_weight_matrix(scaled_series_dict, 'dtw')
    elif similarity_method == 'compute_shape_distance':
        weight_matrix = create_weight_matrix(scaled_series_dict, 'shape')
    elif similarity_method == 'compute_period_distance':
        weight_matrix = create_weight_matrix(scaled_series_dict, 'period')

    arcpy.AddMessage("Similarity matrix for adjacent space-time series constructed!")
    end_time = time.time()
    arcpy.AddMessage(f"Similarity matrix creation completed in {end_time - start_time:.2f} seconds")

    write_weight_to_file(weight_matrix, sts_weight_matrix_csv)

    arcpy.AddMessage(f"Weight matrix data successfully written to {sts_weight_matrix_csv} file!")
    arcpy.AddMessage("----------------------------------")

    ### Regionalization
    arcpy.AddMessage("Starting regionalization process...")
    start_time = time.time()

    # Adjacency matrix
    adj_mx, nodes_set, matrix_len = read_neighbors_file(neighbor_matrix_csv) 

    # Weight matrix
    weight_mx = read_weight_file(sts_weight_matrix_csv, nodes_set, matrix_len)  

    # Create maximum spanning tree
    arcpy.AddMessage("Creating maximum spanning tree...")
    mst_edges = prim_mst(adj_mx, weight_mx, nodes_set)

    arcpy.AddMessage(f"Total nodes: {len(set(u for u, _, _ in mst_edges).union(set(v for _, v, _ in mst_edges)))}")
    arcpy.AddMessage(f"Total edges: {len(mst_edges)}")
    arcpy.AddMessage("Maximum spanning tree created successfully!")
    arcpy.AddMessage("----------------------------------")

    arcpy.AddMessage("Starting tree splitting using genetic algorithm...")
    ga_start_time = time.time()
    """
    mst_edge: Maximum spanning tree
    num_subtree: Number of subtrees to split into
    subtree_size_min: Minimum number of nodes per subtree
    group_size: Number of particles per population
    num_iter: Number of generations to produce
    iter_size: Number of iterations per round
    """

    splitter = UniversalTreeSplitterGA(mst_edges, int(num_subtree), int(subtree_size_min), 
                                     int(group_size), int(num_iter), int(iter_size))

    # Initialize all groups - ensure at least one valid individual
    deletable_edges = splitter.get_deletable_edges()
    arcpy.AddMessage(f"Number of deletable edges: {len(deletable_edges)}")

    subtrees, removed_edges, score = splitter.run_ga()

    if subtrees is None:
        arcpy.AddError(f"Failed to find valid partition. Score: {score}")
    else:
        arcpy.AddMessage("\nOptimal partition solution:")
        for i, edge in enumerate(removed_edges, 1):
            arcpy.AddMessage(f"Removed edge {i}: {edge}")
        arcpy.AddMessage(f"Modularity value: {score}")

        arcpy.AddMessage("\nGenerated subtrees:")
        partition = []
        for i, subtree in enumerate(subtrees, 1):
            nodes = sorted({u for u, _, _ in subtree}.union({v for _, v, _ in subtree}))
            arcpy.AddMessage(f"\nSubtree {i} ({len(nodes)} nodes): {nodes}")
            partition.append(nodes)

        result_dict = {}
        for i, sublist in enumerate(partition):
            result_dict[i+1] = sublist

        inverted_dict = {element: key for key, value_list in result_dict.items() for element in value_list}

        arcpy.AddMessage("Creating output feature class...")
        arcpy.management.CopyFeatures(input_areal_featurelayer, out_feature_class)

        field_name = "subregion_id"
        exist_fields = [f.name for f in arcpy.ListFields(out_feature_class)]
        if field_name not in exist_fields:
            arcpy.management.AddField(out_feature_class, field_name, "LONG")
        
        # Update field values
        with arcpy.da.UpdateCursor(out_feature_class, [area_id_field, field_name]) as cursor:
            update_count = 0
            not_match_count = 0

            for row in cursor:
                node_id_val = row[0]
                if node_id_val in inverted_dict:
                    row[1] = inverted_dict[node_id_val]
                    cursor.updateRow(row)
                    update_count += 1
                else:
                    not_match_count += 1

        del cursor
        arcpy.AddMessage(f"Updated {update_count} records")
        if not_match_count > 0:
            arcpy.AddMessage(f"{not_match_count} records not found in dictionary")
        arcpy.AddMessage("Processing completed")

        # Apply unique value renderer
        apply_unique_value_renderer_complete(out_feature_class, field_name, color_ramp_name="Paired")

        ga_end_time = time.time()
        arcpy.AddMessage(f"Genetic algorithm completed in {ga_end_time - ga_start_time:.2f} seconds")

    arcpy.AddMessage("----------------------------------")

    # Clean up temporary files
    if os.path.exists(neighbor_matrix_csv):
        os.remove(neighbor_matrix_csv)
        arcpy.AddMessage(f"Temporary file {neighbor_matrix_csv} deleted!")

    if os.path.exists(sts_csv):
        os.remove(sts_csv)
        arcpy.AddMessage(f"Temporary file {sts_csv} deleted!")

    if os.path.exists(sts_weight_matrix_csv):
        os.remove(sts_weight_matrix_csv)
        arcpy.AddMessage(f"Temporary file {sts_weight_matrix_csv} deleted!")
    
    end_time_total = time.time()
    arcpy.AddMessage(f"Total processing time: {end_time_total - start_time_total:.2f} seconds")
    return


if __name__ == "__main__":
    # Import needed at runtime to avoid startup issues
    import random

    # Get tool parameters
    input_areal_featurelayer = arcpy.GetParameterAsText(0)
    area_id_field = arcpy.GetParameterAsText(1)
    input_sts_table = arcpy.GetParameterAsText(2)
    area_id = arcpy.GetParameterAsText(3)
    value = arcpy.GetParameterAsText(4)
    time_series = arcpy.GetParameterAsText(5)
    sequence = arcpy.GetParameterAsText(6)
    num_subtree = arcpy.GetParameterAsText(7)
    subtree_size_min = arcpy.GetParameterAsText(8)
    similarity_method = arcpy.GetParameterAsText(9)
    out_feature_class = arcpy.GetParameterAsText(10)
    group_size = arcpy.GetParameterAsText(11)
    num_iter = arcpy.GetParameterAsText(12)
    iter_size = arcpy.GetParameterAsText(13)

    # Run tool
    script_tool(input_areal_featurelayer, area_id_field, input_sts_table, area_id, 
               value, time_series, sequence, num_subtree, subtree_size_min, 
               similarity_method, out_feature_class, group_size, num_iter, iter_size)