#!.venv/bin/python3

import collections.abc
import sys

import itertools
from copy import deepcopy, copy
from functools import cmp_to_key
from typing import List, Tuple, Dict
from instance import Instance, Op
from dataclasses import dataclass, field
from pyscipopt import Model, quicksum, Variable, Constraint

from utils import model_and

# DEFAULT_DATA = 'data/testing/headway1.json'
DEFAULT_DATA = 'data/phase1/line1_critical_0.json'

Var = Dict[Tuple[int, int], Variable]
Values = Dict[Tuple[int, int], int]

@dataclass
class Res_use:
	op: Op
	succ: Op|None
	time: int

class Solution:
	inst: Instance
	paths: List[List[Op]]
	res_uses: Dict[int, List[Res_use]]
	obj_val: int

	def __init__(self, inst: Instance, paths=None, res_uses=None, obj_val=None):
		self.inst = inst
		self.paths = paths
		self.res_uses = res_uses
		self.obj_val = obj_val

		if paths is None:
			self.get_fork_sol()

		if obj_val is None:
			self.get_base_sol()

	@staticmethod
	def get_values(m: Model, var: Var|Tuple[Var], to_int=True) -> Values|Tuple[Values]:
		if isinstance(var, (list, tuple)):
			return tuple(Solution.get_values(m, v, to_int) for v in var)
		
		if to_int:
			rv =  { k: int(round(m.getVal(v))) for k, v in var.items() }
		else:
			rv =  { k: m.getVal(v) for k, v in var.items() }

		return rv 


	def make_obj(self, m: Model, s: Var) -> None:
		o = {}
		c = {}

		for t_ops in self.inst.ops:
			for op in t_ops:
				if op.obj and op in s:
					o[op] = m.addVar(name=f'o{op.idx}', vtype='C', lb=0, ub=None)
					c[op] = op.obj.coeff

					m.addCons(name=f'obj{op.idx}', cons=o[op] >= s[op] - op.obj.threshold)

		m.setObjective(quicksum(c[k]*o[k] for k in o.keys()), sense='minimize')

		return o
	
	def make_col_obj(self, m: Model, u: Var):
		o = {}

		for res, ops in inst.res_ops.items():
			o[res] = m.addVar(name=f'o{res}', vtype='C', lb=0, ub=None)
			cons = quicksum(u[op] for op in ops) - len(ops)/2 <= o[res]
			m.addCons(name=f'obj{res}', cons=cons)

		m.setObjective(quicksum(o[k] for k in o.keys()), sense='minimize')



	@staticmethod
	def make_ops_var(m: Model, ops: List[Op], var_name: str='s', force_use=True, var=None) -> Var:
		assert(var_name in ['s', 'u'])

		if var is None:
			var = {}

		for op in ops:
			if var_name == 's':
				var[op] = m.addVar(name=f's{op.idx}', vtype='C', lb=op.start_lb, ub=op.start_ub)
			elif var_name == 'u':
				var[op] = m.addVar(name=f'u{op.idx}', vtype='B')
				if force_use and (op.n_prev == 0 or op.n_succ == 0 or op.obj):
					m.addCons(name=f'force_u{op.idx}', cons=(var[op] == 1))

		return var


	def make_fork_cons(self, m: Model, u: Var, s: Var, op: Op) -> List[Constraint]:
		rv = []

		# if op.n_succ == 1:
		# 	succ = op.succ[0]

		# 	if succ.n_prev == 1:
		# 		cons = u[op] == u[succ]
		# 	else:
		# 		cons = u[op] <= u[succ]
	
		# 	rv.append(m.addCons(name=f'cont{op.idx}', cons=cons))
		
		if op.n_succ == 1:
			succ = op.succ[0]

			if succ.n_prev == 1:
				cons = u[op] == u[succ]
			else:
				cons = u[op] <= u[succ]
	
			rv.append(m.addCons(name=f'cont{op.idx}', cons=cons))
	
		
		elif op.n_succ > 1:
			cons_lb = quicksum(u[succ] for succ in op.succ) >= u[op]
			cons_ub = quicksum(u[succ] for succ in op.succ) <= 1

			rv.append(m.addCons(name=f'fork_lb{op.idx}', cons=cons_lb))
			rv.append(m.addCons(name=f'fork_ub{op.idx}', cons=cons_ub))

		for succ in op.succ:
			cons = s[op] + op.dur <= s[succ]

			rv.append(m.addConsIndicator(name=f'dur{op.idx},{succ.idx}', 
				cons=cons, binvar=model_and(m, [u[op], u[succ]])))

		return rv
	

	def make_forks_model(self):
		m = Model()

		u = self.make_ops_var(m, itertools.chain(*self.inst.ops), 'u')
		s = self.make_ops_var(m, itertools.chain(*self.inst.ops), 's')
		
		for op in itertools.chain(*self.inst.ops):
			self.make_fork_cons(m, u, s, op)

		self.make_col_obj(m, u)

		return m, (u, s)


	def make_path(self, u: Values, train: int|None=None) -> List[Op]|List[List[Op]]:
		if train is None:
			return [self.make_path(u, i) for i in range(self.inst.n_trains)]
			
		op = self.inst.ops[train][0]
		assert(op.n_prev == 0)
		path = []

		while op.n_succ > 0:
			path.append(op)

			succ = [x for x in op.succ if u[x] == 1]
			assert(len(succ) == 1)

			op = succ[0]

		path.append(op)

		return path

	@staticmethod
	def get_transitions(path: List):
		n = len(path)

		for i in range(n-1):
			yield path[i], path[i+1]

	def make_res_uses(self, path: List, res_uses: Dict[int, List[Res_use]]):
		for op, succ in self.get_transitions(path):
			for res in op.res:
				if not res.idx in res_uses:
					res_uses[res.idx] = []

				res_uses[res.idx].append(Res_use(
					op=op,
					succ=succ,
					time=res.time
				))


	def sort_res_uses(self, s: Values, res_uses: Dict[int, List[Res_use]]|None=None):
		if res_uses is None:
			res_uses = self.res_uses

		def res_use_cmp(x: Res_use, y: Res_use):
			assert(not (x.succ is None and y.succ is None))
			if x.succ is None:
				return False
			
			if y.succ is None:
				return True
			
			if s[x.op] == s[y.op]:
				return s[x.succ] < s[y.succ]
			
			return s[x.op] < s[y.op]

		for v in res_uses.values():
			v.sort(key=cmp_to_key(res_use_cmp))

	@staticmethod
	def make_path_cons(m: Model, s: Var, path: List[Op]):
		rv = []

		for op, succ in zip(path[:-1], path[1:]):
			cons = s[op] + op.dur <= s[succ]
			rv.append(m.addCons(name=f'dur{op.idx}', cons=cons))

		return rv

	@staticmethod
	def make_res_use_cons(m: Model, s: Var, res_uses: List[Res_use]):
		rv = []

		for first, second in zip(res_uses[:-1], res_uses[1:]):
			cons = s[first.succ] + first.time <= s[second.op]
			rv.append(m.addCons(name=f'r{first.op.idx},{second.op.idx}',cons=cons))

		return rv

	def make_base_model(self):
		m = Model()
		
		s = self.make_ops_var(m, itertools.chain(*self.paths))

		for p in self.paths:
			self.make_path_cons(m, s, p)

		for ru in self.res_uses.values():
			self.make_res_use_cons(m, s, ru)

		self.make_obj(m, s)

		return m, s
	
	def get_fork_sol(self):
		mf, (u, s) = self.make_forks_model()
		
		# mf.hideOutput()
		print('forks - optimizing')
		mf.setParam('presolving/maxrounds', 20)
		mf.setParam('limits/stallnodes', 1000)

		mf.optimize()
		print('forks', mf.getObjVal())

		u_v, s_v = self.get_values(mf, (u, s))

		self.paths = self.make_path(u_v)
		self.res_uses = {}
		for p in self.paths:
			self.make_res_uses(p, self.res_uses)
		self.sort_res_uses(s_v)


	def get_base_sol(self):

		mb, _ = self.make_base_model()

		mb.hideOutput()
		print('base - optimizing')
		mb.optimize()
		print('base', mb.getObjVal())

		self.obj_val = mb.getObjVal()
		

	def make_res_improve_cons(self, m: Model, u: Var, s: Var, r: Var,
						   	  res_uses: Dict[int, List[Res_use]], op: Op):
		
		for res in op.res:
			if not res.idx in res_uses:
				continue

			choices = []
			for i, (first, second) in enumerate(zip([None] + res_uses[res.idx], res_uses[res.idx] + [None])):
				k = (res.idx, op.idx, i) 
				r[k] = m.addVar(name=f'r{res.idx},{op.idx},{i}', vtype='B')
				choices.append(r[k])

				if first:
					cons = s[first.succ] + first.time <= s[op]
					
					m.addConsIndicator(name=f'r{first.op.idx},{op.idx}', cons=cons, binvar=r[k])

				if second:
					for succ in op.succ:
						cons = s[succ] + res.time <= s[second.op]

						m.addConsIndicator(name=f'r{op.idx},{second.op.idx},{i}', cons=cons, binvar=r[k])
				
				cons = quicksum(choices) == u[op]
				m.addCons(name=f'ch{op.idx},{res.idx}', cons=cons)
	

	def get_filtered_res_uses(self, train: int):
		return { k: [u for u in v if u.op.train_idx != train]
			for k, v in self.res_uses.items() }

	
	def make_improve_model(self, train: int, res_uses: Dict[int, List[Res_use]]|None=None):
		m = Model()
		
		if res_uses is None:
			res_uses = self.get_filtered_res_uses(train)
		
		u = {}
		s = {}
		r = {}

		for i, p in enumerate(self.paths):
			if i == train:
				continue

			self.make_ops_var(m, p, 's', var=s)
			self.make_path_cons(m, s, p)

		

		for ru in res_uses.values():
			self.make_res_use_cons(m, s, ru)

		self.make_ops_var(m, self.inst.ops[train], 's', var=s)
		self.make_ops_var(m, self.inst.ops[train], 'u', var=u)

		for op in self.inst.ops[train]:
			self.make_fork_cons(m, u, s, op)
			self.make_res_improve_cons(m, u, s, r, res_uses, op)

		self.make_obj(m, s)

		return m, (u, s, r)
	

	def get_improve_sol(self, train: int):
		filt_res_uses = self.get_filtered_res_uses(train)
		m, (u, s, r) = self.make_improve_model(train, filt_res_uses)

		m.hideOutput()
		print(f'impr {train} - optimizing')
		m.setParam('limits/stallnodes', 1000)
		m.optimize()
		print(f'impr {train}', m.getObjVal())

		u_v, s_v = sol.get_values(m, (u, s))

		new_path = self.make_path(u_v, train)

		imp_paths = [[op for op in p] if i != train else new_path for i, p in enumerate(self.paths)]
		
		imp_res_uses = {}
		for p in imp_paths:
			self.make_res_uses(p, imp_res_uses)
		self.sort_res_uses(s_v, imp_res_uses)
		
		imp_sol = Solution(self.inst, imp_paths, imp_res_uses)
	
		# print('old path', self.paths[train])
		# print('new path', new_path)

		# print(new_res_uses)

		return imp_sol
	
if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solution(inst)

	
	while True:
		best = sol
		for tr in range(inst.n_trains):
			ns = sol.get_improve_sol(tr)
			if ns.obj_val < best.obj_val:
				best = ns
		
		if best.obj_val >= sol.obj_val:
			break
		
		sol = best
			



