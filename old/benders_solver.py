#!.venv/bin/python3

import sys

import gurobipy as gp
import networkx as nx

from instance import Instance, Op
from gurobipy import GRB

from collections import deque

DEFAULT_DATA = 'data/testing/headway1.json'
# DEFAUL_DATA = 'data/phase1/line1_critical_0.json'


class Master_problem:
	gm: gp.Model
	inst: Instance

	def __init__(self, inst):
		self.inst = inst
		self.create_model()

	def create_op_vars(self):
		m = self.gm
		self.var_op = m.addVars(self.inst.all_ops, name='op', vtype='B')

	def create_point_flow_conss(self):
		m = self.gm
		var_op = self.var_op

		for p in inst.all_points:
			if not p.ops_in:
				m.addConstr(gp.quicksum(var_op[op] for op in p.ops_out) == 1)
			elif not p.ops_out:
				m.addConstr(gp.quicksum(var_op[op] for op in p.ops_in) == 1)
			else:
				m.addConstr(
					gp.quicksum(var_op[op] for op in p.ops_in) == 
					gp.quicksum(var_op[op] for op in p.ops_out))

	def create_res_vars(self):
		self.locks = {}
		self.unlocks = {}

		for p in inst.all_points:
			for r in p.res_lock.keys():
				self.locks.setdefault(r, [None]).append(p)

			for r in p.res_unlock.keys():
				self.unlocks.setdefault(r, [None]).append(p)

		m = self.gm

		keys = []
		for r in self.locks.keys():
			def keys_filt_func(p_unlock, p_lock):
				if p_unlock is None or p_lock is None:
					return True
				
				if p_unlock.train_idx == p_lock.train_idx:
					if p_unlock.point_idx >= p_lock.point_idx:
						return False
					
				return True
			
			keys += [(r, p_unlock, p_lock) 
				for p_unlock in self.unlocks[r]
				for p_lock in self.locks[r]
				if keys_filt_func(p_unlock, p_lock)]
			
		
		self.var_res = m.addVars(keys, name='res', vtype='B')


	def create_res_conss(self):
		m = self.gm
		var_op = self.var_op
		var_res = self.var_res

		for p in self.inst.all_points:
			for r, ops in p.res_unlock.items():
				m.addConstr(gp.quicksum(var_op[op] for op in ops) == var_res.sum(r, p, '*'))
				if len(p.ops_out) == 0:
					m.addConstr(gp.quicksum(var_op[op] for op in ops) == var_res[r, p, None])

			
			for r, ops in p.res_lock.items():
				m.addConstr(gp.quicksum(var_op[op] for op in ops) == var_res.sum(r, '*', p))
				if len(p.ops_in) == 0:
					m.addConstr(gp.quicksum(var_op[op] for op in ops) == var_res[r, None, p])

		for r in self.locks.keys():
			m.addConstr(var_res.sum(r, None, '*') == 1)
			m.addConstr(var_res.sum(r, '*', None) == 1)


	def create_model(self):
		self.gm = gp.Model('train opt')
		
		self.create_op_vars()
		self.create_point_flow_conss()
		self.create_res_vars()
		self.create_res_conss()

	def optimize(self, callback=None):
		print('optimizing')
		if callback:
			self.gm.Params.LazyConstraints = 1
		self.gm.optimize(callback)

class Cycle_callback:
	def __init__(self, model, inst):
		self.model = model
		self.inst = inst
		self.n_cycle_conss = 0

	def __call__(self, model, where):
		if where == GRB.Callback.MIPSOL:
			self.remove_cycles(model)

	def remove_cycles(self, model):
		val_op = model.cbGetSolution(self.model.var_op)
		val_res = model.cbGetSolution(self.model.var_res)

		
		edges = { (op.p_start, op.p_end): self.model.var_op[op] 
			   for op, v in val_op.items() if v > 0.5 }
		
		edges.update({ (p1, p2): self.model.var_res[r, p1, p2]
			   for (r, p1, p2), v in val_res.items() 
			   if v > 0.5 and not (p1 is None or p2 is None)})
	
		g = nx.DiGraph()
		g.add_edges_from(edges.keys())

		scc = [x for x in nx.strongly_connected_components(g) if len(x) > 1]

		if len(scc) == 0:
			print('found solution')
			model.terminate()

		for c in scc:
			g_c = nx.subgraph(g, c)
			sc = self.find_shortest_cycle(g_c)
			g_sc = nx.subgraph(g, sc)
			self.model.gm.cbLazy(gp.quicksum(edges[e] for e in g_sc.edges) <= len(g_sc) - 1)
			print('eliminating:', sc)


	@staticmethod
	def find_shortest_cycle(g):
		adj = nx.to_numpy_array(g)

		nodes = list(g.nodes)
		prev = { (int(i), int(j)): int(i) for i, j in zip(*adj.nonzero()) }
		# print(prev)

		n = len(nodes)

		def reconstruct_cycle(s):
			path = [s]

			k = s
			while True:
				k = prev[s, k]
				if k == s:
					break
				path.insert(0, k)

			path = [nodes[x] for x in path]
			return path


		for k in range(n):
			for i in range(n):
				for j in range(n):
					if adj[i, j]:
						continue

					if adj[i, k] and adj[k, j]:
						adj[i, j] = 1
						prev[i, j] = prev[k, j]

						if i == j:
							return reconstruct_cycle(i)
		
		return None


		
class Solver:
	inst: Instance
	def __init__(self, inst):
		self.inst = inst


	def solve(self):
		mp = Master_problem(inst)

		cb = Cycle_callback(mp, self.inst)

		mp.optimize(cb)		
		

if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solver(inst)
	sol.solve()
