from collections import Counter
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx


def get_y(y, idx_train):
	return [y[index].item() for index in idx_train]

def plot_splits(y, idx_train, idx_val, idx_test):

	counter = Counter(y)

	count = [x[1] for x in sorted(counter.items())]

	fig = make_subplots(rows=1, cols=3, shared_yaxes=True)
	idx_list = [idx_train, idx_val, idx_test]
	plot_names = ['train', 'val', 'test']

	for i in range(3):

		idxs = idx_list[i]
		y_temp = get_y(y, idxs)

		counter = Counter(y_temp)

		count = [x[1] for x in sorted(counter.items())]

		bar_plot_trace = go.Bar(
			x=list(range(7)),
			y=count,
			text=count,
			name=plot_names[i],
			textposition='outside'
		)

		fig.add_trace(
			bar_plot_trace,
			row=1,
			col=i+1
		)


	fig.update_layout(
		width=1200,
		height=600, 
		yaxis_title="Counts",
		yaxis_title_font=dict(size=15),
		title={
		'text': "Train/Val/Test Class Makeup",
		'x': 0.5, 
		'y': 0.95,
		'xanchor': 'center',
		'yanchor': 'top',
	},
	)

	fig.show()



def plot_heatplot(counts, label_dict, title):
	heatmap = go.Heatmap(
		z=counts,
		colorscale='Viridis_r', 
		colorbar=dict(title="Scaled Value") 
	)

	fig = go.Figure(data=[heatmap])

	annotations = []
	for i in range(counts.shape[0]):
		for j in range(counts.shape[1]):
			annotations.append(
				dict(
					x=j,
					y=i,
					text=str(f"{counts[i, j]:.2f}"), 
					showarrow=False,
					font=dict(size=12, color="black")
				)
			)

	fig.update_layout(
		title={
			'text':title,
			'x': 0.5,
			'y':0.95,
			'xanchor': 'center',
			'yanchor': 'top'
		},
		xaxis_title="Class",
		yaxis_title="Class",
		xaxis=dict(
			tickmode='array',
			tickvals=list(range(counts.shape[1])),
			ticktext=list(label_dict.values())
		),
		yaxis=dict(
			tickmode='array',
			tickvals=list(range(counts.shape[0])),
			ticktext=list(label_dict.values()),
			autorange='reversed'
		),
		annotations=annotations,
		width=900,
		height=700
	)

	fig.show()

def plot_degrees_histogram(degrees):
	histogram = go.Histogram(x=degrees, nbinsx=50)

	fig = go.Figure(data=[histogram])

	fig.update_layout(
		xaxis_title="Value",
		yaxis_title="Frequency",
		width=800,
		height=600,
		title={
			'text': "Degrees",
			'x': 0.5, 
			'y': 0.95,
			'xanchor': 'center',
			'yanchor': 'top',
		}
	)

	fig.show()



def plot_counts(count):
	bar_plot_trace = go.Bar(
			x=list(range(7)),
			y=count,
			text=count,
			textposition='outside'
	)

	fig = go.Figure(data=[bar_plot_trace])

	fig.update_layout(
		width=1000,
		height=600, 
		xaxis_title="Class",
		xaxis_title_font=dict(size=15),
		yaxis_title="Counts",
		yaxis_title_font=dict(size=15),
		title={
		'text': "Class Counts",
		'x': 0.5, 
		'y': 0.95,
		'xanchor': 'center',
		'yanchor': 'top',
	},
	)

	fig.show()



def plot_eigenvalues(e):
	trace = go.Histogram(x=e, nbinsx=100) 

	fig = go.Figure(data=[trace])

	fig.update_layout(
		xaxis_title="Value",
		yaxis_title="Frequency",
		width=800,
		height=600,
		title={
			'text': "Eigenvalues",
			'x': 0.5, 
			'y': 0.95,
			'xanchor': 'center',
			'yanchor': 'top',
		}
	)

	fig.show()


def plot_graph(data, label_dict):
	G = to_networkx(data, to_undirected=True)
	node_color = []
	nodelist = [[], [], [], [], [], [], []]
	colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
	labels = data.y

	for n, i in enumerate(labels):
		node_color.append(colorlist[i])
		nodelist[i].append(n)

	pos = nx.spring_layout(G, seed = 42)
	plt.figure(figsize = (10, 10))
	labellist = list(label_dict.values())

	for num, i in enumerate(zip(nodelist, labellist)):
		n, l = i[0], i[1]
		nx.draw_networkx_nodes(G, pos, nodelist=n, node_size = 5, node_color = colorlist[num], label=l)
	
	nx.draw_networkx_edges(G, pos, width = 0.25)
	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')


def get_edge_example(data, index:int=0):
	edge_index = data.edge_index.numpy()
	edge_example = edge_index[:, np.where(edge_index[0]==index)[0]]
	return edge_example

def plot_example_nodes(data):
	nrows, ncols = 3, 5
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))
	axs = axs.flatten()

	for i in range(nrows*ncols):

		edge_example = get_edge_example(data, i)

		G = nx.Graph()
		node_example = np.unique(edge_example.flatten())
		G.add_nodes_from(node_example)
		G.add_edges_from(list(zip(edge_example[0], edge_example[1])))
		
		nx.draw_networkx(G, ax=axs[i], with_labels=True)
	
	plt.tight_layout()
	plt.show()
