from collections import defaultdict

class Disjoint_set:
	def __init__(self, n_items):
		self.n_items = n_items
		self.n_sets = n_items

		self.parent = list(range(n_items))
		self.size = [1] * n_items

	def find_set(self, v):
		while (v != self.parent[v]):
			self.parent[v] = self.parent[self.parent[v]]
			v = self.parent[v]
		
		return v
	
	def union_set(self, a, b):
		a = self.find_set(a)
		b = self.find_set(b)

		if (a != b):
			if self.size[a] < self.size[b]:
				a, b = b, a
			
			self.parent[b] = a
			self.size[a] += self.size[b]

			self.n_sets -= 1
	
	def get_sets(self):
		sets = defaultdict(list)

		for v in range(self.n_items):
			sets[self.find_set(v)].append(v)

		return list(sets.values())
