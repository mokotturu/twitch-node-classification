import numpy as np
import networkx as nx
from torch_geometric.datasets import Twitch
from tqdm import tqdm

from multiprocessing import Pool
import time
import itertools


def chunks(l, n):
	"""Divide a list of nodes `l` in `n` chunks"""
	l_c = iter(l)
	while 1:
		x = tuple(itertools.islice(l_c, n))
		if not x:
			return
		yield x


def betweenness_centrality_parallel(G, processes=None):
	"""Parallel betweenness centrality  function"""
	p = Pool(processes=processes)
	node_divisor = len(p._pool) * 4
	node_chunks = list(chunks(G.nodes(), G.order() // node_divisor))
	num_chunks = len(node_chunks)
	bt_sc = p.starmap(
		nx.betweenness_centrality_subset,
		zip(
			[G] * num_chunks,
			node_chunks,
			[list(G)] * num_chunks,
			[True] * num_chunks,
			[None] * num_chunks,
		),
	)

	# Reduce the partial solutions
	bt_c = bt_sc[0]
	for bt in bt_sc[1:]:
		for n in bt:
			bt_c[n] += bt[n]
	return bt_c

langs = [
	'DE',
	'DE',
	'EN',
	'ES',
	'FR',
	'PT',
	'RU',
]

for lang in tqdm(langs):
	data = np.load(f'data/Twitch/{lang}/raw/{lang}.npz')
	G = nx.from_edgelist(data['edges'])

	G.remove_edges_from(nx.selfloop_edges(G))

	# # Clustering coefficient
	# clustering_coefficient = nx.clustering(G)
	# # Degree centrality
	# degree_centrality = nx.degree_centrality(G)

	# # Betweenness centrality
	# betweenness_centrality = nx.betweenness_centrality(G)
	print("")
	print("Computing betweenness centrality for:")
	print(G)
	print("\tParallel version")
	start = time.time()
	bt = betweenness_centrality_parallel(G)
	print(f"\t\tTime: {(time.time() - start):.4F} seconds")
	print(f"\t\tBetweenness centrality for node 0: {bt[0]:.5f}")
	print("")




	# Closeness centrality
	closeness_centrality = nx.closeness_centrality(G)

	# print(lang, betweenness_centrality)