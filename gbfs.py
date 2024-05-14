from queue import PriorityQueue
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

graph_file = 'graph.txt'

def greedy_best_first_search(graph, start_node, goal_node):
    priority_queue = PriorityQueue()
    priority_queue.put((0, start_node, [start_node], 0))  # (priority, node, path, cost)
    
    visited = set()
    visited.add(start_node)
    
    while not priority_queue.empty():
        _, current_node, path, path_cost = priority_queue.get()
        
        if current_node == goal_node:
            return path
        
        try:
            graph[current_node]
        except:
            graph[current_node] = {}

        for neighbor, cost in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                priority = heuristic[neighbor]
                new_path = path + [neighbor]
                new_path_cost = path_cost + cost
                priority_queue.put((priority, neighbor, new_path, new_path_cost))
    
    return None


start_node = 'Tumkur'
goal_node = 'Shivamogga'

graph = {}
heuristic = {}

with open(graph_file, 'r') as f:
    #(node heuristic child1,cost child2,cost )
    for line in f:
        node, *neighbors = line.strip().split()
        edges = []
        heuristic[node]= neighbors[0]
        for neighbor in neighbors[1:]:
            neighbor, cost = neighbor.split(',')
            edges.append((neighbor, int(cost)))
        graph[node] = edges



def findcost(s,d):
    s = graph[s]
    for i in s:
        if (i[0]== d):
            return i[1]
    return -1


print("Graph:")
print(graph)
print("\nHeuristic value:")
print(heuristic,end='\n\n')
plt.rcParams['figure.figsize'] = [8, 6]

def draw_graphs(graph, path,s):
    G = nx.Graph()

    for node, neighbors in graph.items():
        G.add_node(node)
        for neighbor, cost in neighbors:
            G.add_edge(node, neighbor, weight=cost)


    pos = graphviz_layout(G, prog='neato',root=start_node)

    node_labels = {}
    edge_labels = {}

    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=500,edge_color='red')

    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', width=4)
        for i, edge in enumerate(path_edges):
            node_labels[edge[0]] = len(path)
            edge_labels[edge] = findcost(edge[0],edge[1])
                
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12,rotate=0)

    for node, _ in node_labels.items():
        x, y = pos[node]
        plt.text(x, y + 20, heuristic[node], va='center_baseline', color='green')

    plt.title("Solution Graph\nCost = "+s)
    plt.axis('off')
   
    plt.savefig("solution.png")#transparent=True
    plt.show()
    plt.draw()

def draw_graph(graph):
    G = nx.Graph()
    
    for node, neighbors in graph.items():
        G.add_node(node)
        for neighbor, cost in neighbors:
            G.add_edge(node, neighbor, weight=cost)

    node_labels = {}
    path_edges = []
    for node, neighbors in graph.items():
        for neighbor, cost in neighbors:
            path_edges.append((node, neighbor))

    edge_labels = {}
    for i, edge in enumerate(path_edges):
        node_labels[edge[0]] = len(graph) - i - 1
        edge_labels[edge] = findcost(edge[0],edge[1])

    pos = graphviz_layout(G, prog='neato',root=start_node)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12,rotate=0,horizontalalignment='right',verticalalignment='top')
    
    for node, _ in node_labels.items():
        x, y = pos[node]
        plt.text(x, y + 20, heuristic[node], va='center_baseline', color='green')

    nx.draw_networkx_edges(G, pos, width=1)
    
    plt.title("Problem Graph")
    plt.axis('off')
    
    plt.savefig("problem.png")#transparent=True
    plt.show()
    plt.draw()


draw_graph(graph)
path = greedy_best_first_search(graph, start_node, goal_node)


if path:
    cost = 0
    last = start_node
    for i in path[1:]:
        print(last,end='')
        print(" -> ",end='')
        cost = cost + findcost(last,i)
        last = i
    print(goal_node)

    print("Cost is ",end='')
    print(cost)
    draw_graphs(graph, path,str(cost))
else:
    print("Goal node not found.")