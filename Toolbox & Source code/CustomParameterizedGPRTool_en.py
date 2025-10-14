import arcpy
import time
import os
import random
import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity
import heapq

class UniversalTreeSplitterGA:
    """
    Universal Tree Splitter with Hierarchical Genetic Algorithm
    Optimized implementation for regionalization tasks
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
        
        # Cache for fitness values
        self.fitness_cache = {}

    def get_deletable_edges(self):
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

    def run_ga(self):
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
                # Random selection of N-1 edges
                edge_indices = random.sample(deletable_edges, self.num_subtree - 1)
                fitness = self.evaluate_fitness(edge_indices)
                if fitness > -1:  # Only accept valid individuals
                    population.append(edge_indices)
                attempts += 1
                
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
            for group in groups:
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

    def _update_group(self, group, deletable_edges):
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
        
        # Crossover and mutation
        while len(new_population) < len(population):
            # Select parents with probability proportional to fitness
            weights = np.array(fitnesses) / sum(fitnesses)
            parent1_index, parent2_index = np.random.choice(range(len(population)), p=weights, size=2, replace=False)
            parent1, parent2 = population[parent1_index], population[parent2_index]
            fitness1, fitness2 = fitnesses[parent1_index], fitnesses[parent2_index]
            
            # Crossover
            if len(parent1) <= 2:  # Skip crossover for individuals with very few genes
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

    def evaluate_fitness(self, individual):
        """
        Evaluate individual fitness (higher is better) with caching
        
        Parameters:
            individual -- List of edge indices to remove
            
        Returns:
            Fitness value (modularity score, or negative value if invalid)
        """
        try:
            # Convert to hashable type for caching
            individual_key = tuple(sorted(individual))
            
            # Check cache first
            if individual_key in self.fitness_cache:
                return self.fitness_cache[individual_key]
                
            edge_indices = [int(x) for x in individual]
            temp_G = self.G.copy()
            for idx in edge_indices:
                edge = self.edges[idx]
                temp_G.remove_edge(*edge)
                
            components = list(nx.connected_components(temp_G))
            cc_list = [list(c) for c in components]
            Q = round(modularity(self.G, cc_list), 4)
            fitness = max(Q, 0.00001)  # Ensure return value is a small positive number
            
            # Store in cache
            self.fitness_cache[individual_key] = fitness
            return fitness
        except:
            return -10

    def crossover(self, parent1, fitness1, parent2, fitness2):
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
                # Update child with parent genes
                needed = size - len(child)
                available = list(set(parent) - child)
                
                if needed <= len(available):
                    child.update(random.sample(available, k=needed))
                else:
                    child.update(available)
                    # If still not enough, use genes from other sources
                    child.update(random.sample(list(set(range(self.num_edges)) - child), k=needed - len(available)))
                
                return list(child)

            child1 = fill_child(child1, parent2)
            child2 = fill_child(child2, parent1)

            # Evaluate fitness
            fit_child1 = self.evaluate_fitness(child1)
            fit_child2 = self.evaluate_fitness(child2)
            
            # Return best combinations
            if fit_child1 > fitness1 and fit_child2 > fitness2:
                return child1, fit_child1, child2, fit_child2
            elif fit_child1 > fitness1:
                return child1, fit_child1, parent2, fitness2
            elif fit_child2 > fitness2:
                return parent1, fitness1, child2, fit_child2
                
        return parent1, fitness1, parent2, fitness2

    def mutate(self, individual, fit_individual, deletable_edges):
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
            # Mutate by replacing one or more edges
            num_mutations = random.randint(1, max(1, int(len(mutated) * 0.3)))  # Mutate up to 30% of genes
            
            for _ in range(num_mutations):
                idx = random.randint(0, len(mutated) - 1)
                # Ensure we pick a different edge than what's already there
                current_edge = mutated[idx]
                available_edges = [e for e in deletable_edges if e != current_edge]
                if available_edges:
                    mutated[idx] = random.choice(available_edges)
            
            mutated_fitness = self.evaluate_fitness(mutated)
            if mutated_fitness > fit_individual:
                return mutated, mutated_fitness
                
        return individual, fit_individual

    def _build_result(self, edge_indices):
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
            comp_set = set(comp)  # Convert to set for O(1) lookups
            
            for u, v, w in self.mst_edges:
                if u in comp_set and v in comp_set:
                    subtree.append((u, v, w))

            # Handle isolated nodes
            if not subtree and comp_set:
                node = next(iter(comp_set))  # Get the single node
                subtree.append((node, node, -1))  # Mark isolated node

            subtrees.append(subtree)

        return subtrees, removed_edges


def read_neighbors_file(file_path):
    """
    Read neighbor file and generate adjacency matrix
    
    Parameters:
        file_path (str): Path to neighbor file
        
    Returns:
        tuple: (adjacency matrix, set of nodes, matrix size)
    """
    arcpy.AddMessage(f"Reading neighbor information from {file_path}...")
    
    with open(file_path, 'r') as f:
        # Initialize neighbor dictionary
        neighbors_dict = {}
        max_node = 0
        nodes_set = set()
        lines_processed = 0
        
        # Read all lines and process
        for line in f:
            nodes = list(map(int, line.strip().split(',')))
            if not nodes:
                continue
                
            current_node = nodes[0]
            neighbors = nodes[1:] if len(nodes) > 1 else []
            
            # Update maximum node ID
            max_node = max(max(nodes) if nodes else 0, max_node)
            # Count all nodes
            nodes_set.add(current_node)
            # Store neighbor relationships
            neighbors_dict[current_node] = neighbors
            lines_processed += 1
            
    arcpy.AddMessage(f"Processed {lines_processed} neighbor records")

    # Determine total number of nodes
    matrix_len = max_node + 1
    
    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((matrix_len, matrix_len), dtype=int)

    # Fill adjacency matrix
    arcpy.AddMessage("Building adjacency matrix...")
    for node, neighbors in neighbors_dict.items():
        for neighbor in neighbors:
            adjacency_matrix[node, neighbor] = 1
            adjacency_matrix[neighbor, node] = 1  # Undirected graph needs to be symmetric

    arcpy.AddMessage(f"Created adjacency matrix of size {matrix_len}x{matrix_len} with {len(nodes_set)} active nodes")
    return adjacency_matrix, nodes_set, matrix_len


def read_weight_file(file_path, nodes_size, matrix_len):
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

        # More efficient dictionary storage for edges
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
        
        # Process edges efficiently
        edges_processed = 0
        
        # Set up progress reporting
        total_nodes = len(nodes_size)
        arcpy.SetProgressor("step", "Processing weight matrix...", 0, total_nodes, 1)
        
        # Convert nodes_size to a list for deterministic order
        node_list = list(nodes_size)
        
        for i, node_i in enumerate(node_list):
            for node_j in node_list:
                if node_i == node_j:
                    continue
                
                # Get weights for both directions
                weight_ij = edge_dict.get((node_i, node_j), 0)
                weight_ji = edge_dict.get((node_j, node_i), 0)

                # Undirected graph weight is sum of both directions
                total_weight = weight_ij + weight_ji

                if total_weight > 0:
                    weight_matrix[node_i, node_j] = total_weight
                    edges_processed += 1
            
            # Update progress
            arcpy.SetProgressorPosition(i + 1)
            if (i + 1) % max(1, total_nodes // 10) == 0:  # Report every 10%
                arcpy.AddMessage(f"Processed {i + 1} of {total_nodes} nodes ({((i + 1) / total_nodes) * 100:.1f}%)")

        arcpy.ResetProgressor()
        arcpy.AddMessage(f"Created weight matrix with {edges_processed} weighted edges")
        return weight_matrix


def prim_mst(adj_matrix, weight_matrix, nodes_set):
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
    
    # Set up progress reporting
    total_nodes = len(nodes_set)
    arcpy.SetProgressor("step", "Building maximum spanning tree...", 0, total_nodes, 1)
    
    while unvisited:
        component_count += 1
        # Randomly select an unvisited node as starting point
        start_node = random.choice(list(unvisited))
        visited = {start_node}
        mst_edges = []
        candidate_edges = []  # Candidate edge heap
        
        # Initialize candidate edges (all edges from starting node)
        for j in range(n):
            if adj_matrix[start_node, j] == 1:
                weight = float(weight_matrix[start_node, j])
                heapq.heappush(candidate_edges, (-weight, start_node, j))  # Use negative values for max heap
        
        # Main Prim's algorithm loop with heap optimization
        while candidate_edges:
            neg_weight, i, j = heapq.heappop(candidate_edges)  # O(log n) operation
            weight = -neg_weight
            
            if j not in visited:
                mst_edges.append((i, j, weight))
                visited.add(j)
                
                # Add new node's edges to candidates
                for k in range(n):
                    if adj_matrix[j, k] == 1 and k not in visited:
                        new_weight = float(weight_matrix[j, k])
                        heapq.heappush(candidate_edges, (-new_weight, j, k))
        
        # Record current component's spanning tree
        all_msts.append(mst_edges)
        unvisited -= visited
        
        # Update progress
        nodes_processed = total_nodes - len(unvisited)
        arcpy.SetProgressorPosition(nodes_processed)
        arcpy.AddMessage(f"Component {component_count}: {len(visited)} nodes, {len(mst_edges)} edges")

    arcpy.ResetProgressor()
    
    # Find spanning tree with most nodes
    largest_mst = max(all_msts, key=lambda mst: len(set(u for u, v, _ in mst).union(set(v for _, v, _ in mst))))
    
    arcpy.AddMessage(f"Found maximum spanning tree with {len(largest_mst)} edges")
    return largest_mst


def apply_unique_value_renderer_complete(output_fc, field_name="subregion_id", 
                                       color_ramp_name="Prismatic"):
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


def script_tool(input_areal_featurelayer, area_id_field, neighbor_matrix_csv, 
                sts_weight_matrix_csv, num_subtree, subtree_size_min, out_feature_class, 
                group_size, num_iter, iter_size):
    """
    Main script tool function
    
    Parameters:
        input_areal_featurelayer: Input areal feature layer
        area_id_field: Area ID field
        neighbor_matrix_csv: Path to neighbor matrix CSV
        sts_weight_matrix_csv: Path to weight matrix CSV
        num_subtree: Number of subtrees to create
        subtree_size_min: Minimum size of each subtree
        out_feature_class: Output feature class
        group_size: Population size for genetic algorithm
        num_iter: Number of iterations for genetic algorithm
        iter_size: Size of each iteration
    """
    arcpy.AddMessage("Starting regionalization process...")
    total_start_time = time.time()

    # Read adjacency matrix
    adj_mx, nodes_set, matrix_len = read_neighbors_file(neighbor_matrix_csv) 

    # Read weight matrix
    weight_mx = read_weight_file(sts_weight_matrix_csv, nodes_set, matrix_len)  

    # Create maximum spanning tree
    arcpy.AddMessage("Creating maximum spanning tree...")
    start_time = time.time()
    mst_edges = prim_mst(adj_mx, weight_mx, nodes_set)

    arcpy.AddMessage(f"Total nodes: {len(set(u for u, _, _ in mst_edges).union(set(v for _, v, _ in mst_edges)))}")
    arcpy.AddMessage(f"Total edges: {len(mst_edges)}")
    arcpy.AddMessage(f"Maximum spanning tree created successfully in {time.time() - start_time:.2f} seconds!")
    arcpy.AddMessage("----------------------------------")

    arcpy.AddMessage("Starting tree splitting using genetic algorithm...")
    ga_start_time = time.time()

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
        
        # Update field values with progress reporting
        arcpy.AddMessage("Updating feature attributes...")
        row_count = int(arcpy.GetCount_management(out_feature_class).getOutput(0))
        arcpy.SetProgressor("step", "Updating subregion values...", 0, row_count, 1)
        
        with arcpy.da.UpdateCursor(out_feature_class, [area_id_field, field_name]) as cursor:
            update_count = 0
            not_match_count = 0
            i = 0

            for row in cursor:
                node_id_val = row[0]
                if node_id_val in inverted_dict:
                    row[1] = inverted_dict[node_id_val]
                    cursor.updateRow(row)
                    update_count += 1
                else:
                    not_match_count += 1
                
                i += 1
                arcpy.SetProgressorPosition(i)
                if i % max(1, row_count // 10) == 0:  # Report every 10%
                    arcpy.AddMessage(f"Processed {i} of {row_count} features ({(i/row_count*100):.1f}%)")

        arcpy.ResetProgressor()
        arcpy.AddMessage(f"Updated {update_count} records")
        if not_match_count > 0:
            arcpy.AddMessage(f"{not_match_count} records not found in dictionary")
        arcpy.AddMessage("Processing completed")

        # Apply unique value renderer
        apply_unique_value_renderer_complete(out_feature_class, field_name, color_ramp_name="Paired")

        ga_end_time = time.time()
        arcpy.AddMessage(f"Genetic algorithm completed in {ga_end_time - ga_start_time:.2f} seconds")

    arcpy.AddMessage("----------------------------------")
    total_end_time = time.time()
    arcpy.AddMessage(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")
    return


if __name__ == "__main__":
    # Get tool parameters
    input_areal_featurelayer = arcpy.GetParameterAsText(0)
    area_id_field = arcpy.GetParameterAsText(1)
    neighbor_matrix_csv = arcpy.GetParameterAsText(2)
    sts_weight_matrix_csv = arcpy.GetParameterAsText(3)
    num_subtree = arcpy.GetParameterAsText(4)
    subtree_size_min = arcpy.GetParameterAsText(5)
    out_feature_class = arcpy.GetParameterAsText(6)
    group_size = arcpy.GetParameterAsText(7)
    num_iter = arcpy.GetParameterAsText(8)
    iter_size = arcpy.GetParameterAsText(9)

    # Run tool
    script_tool(input_areal_featurelayer, area_id_field, neighbor_matrix_csv, 
               sts_weight_matrix_csv, num_subtree, subtree_size_min, 
               out_feature_class, group_size, num_iter, iter_size)