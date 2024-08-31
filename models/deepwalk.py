import random
from gensim.models import Word2Vec

class DeepWalk():

	def __init__(self, G, num_walks, length_walk, window_size, embedding_size):
		self.G = G
		self.num_walks = num_walks
		self.length_walk = length_walk
		self.window_size = window_size
		self.embedding_size = embedding_size
		self.workers = 4
	
	def random_walk(self):

		nodes = self.G.nodes()
		walks = []

		for node in nodes:
			walk = [str(node)]
			
			for _ in range(self.length_walk):
				nbs = list(self.G.neighbors(node))
				if len(nbs) == 0:
					break
				node = random.choice(nbs)
				walk.append(str(node))
			
			walks.append(walk)

		return walks


	def sentences(self):
		sts = []
		for _ in range(self.num_walks):
			sts.extend(self.random_walk())

		return sts
	

	def train(self):
		walks = self.sentences()
		walks = [[str(node) for node in walk] for walk in walks]  # Convert nodes to strings for gensim Word2Vec
		model = Word2Vec(walks, vector_size=self.embedding_size, window=self.window_size, workers=self.workers, sg=1)
		
		return model