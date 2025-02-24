#!.venv/bin/python3

import sys

from typing import List, Tuple, Dict
from itertools import chain
from instance import Instance, Op
from dataclasses import dataclass, field
from pyscipopt import Model, quicksum, Variable

# DEFAUL_DATA = 'data/testing/headway1.json'
DEFAUL_DATA = 'data/phase1/line1_critical_0.json'


@dataclass
class Res_use:
	op: Op
	succ: int|None
	time: int
	start: int
	end: int|None


@dataclass
class Res_col:
	res: int
	op1: Op
	succ1: Op
	time1: int
	op2: Op
	succ2: Op
	time2: int

@dataclass
class Res_order:
	first: Op
	second: Op
	time: int

class Solver:
	inst: Instance

	def __init__(self, inst):
		self.inst = inst 
	
	def get_res_overlap(self):
		self.res_op = {}
		for train_ops in self.inst.ops:
			for op in train_ops:
				for res in op.res:
					if not res.idx in self.res_op:
						self.res_op[res.idx] = []
					self.res_op[res.idx].append((op, res.time))

	def get_end_ops(self):
		return [[op for op in train_ops if op.n_succ == 0] for train_ops in self.inst.ops]
	
	def get_start_ops(self):
		return [[op for op in train_ops if op.n_prev == 0] for train_ops in self.inst.ops]

	def make_model_obj(self, m, ops, start):
		obj = {}
		obj_c = {}
		for train_ops in ops:
			for op in train_ops:
				if op.obj is not None:
					obj[op.idx] = m.addVar(name=f'obj{op.idx}', vtype='C', lb=0, ub=None)
					m.addCons(name=f'obj{op.idx}', 
						cons=obj[op.idx] >= start[op.idx] - op.obj.threshold)

					obj_c[op.idx] = op.obj.coeff
		
		m.setObjective(quicksum(obj_c[k]*obj[k] for k in obj.keys()), sense='minimize')
	
	def make_forks_model(self):
		m = Model()
		m.setProbName('forks')

		used = {}
		start = {}

		omega = self.inst.total_dur
		
		for train_ops in self.inst.ops:
			for op in train_ops:
				used[op.idx] = m.addVar(name=f'used{op.idx}', vtype='B')
				start[op.idx] = m.addVar(name=f'start{op.idx}', vtype='C', 
										 lb=op.start_lb, ub=op.start_ub)

				if op.n_prev == 0 or op.n_succ == 0 or op.obj is not None:
					m.addCons(used[op.idx] == 1)

			for op in train_ops:
				if op.n_succ == 1:
					m.addCons(name=f'cont{op.idx}',
			   			cons=(used[op.succ[0].idx] >= used[op.idx]))
					
				elif op.n_succ > 1:
					m.addCons(name=f'fork_lb{op.idx}',
						cons=(quicksum(used[succ.idx] for succ in op.succ) >= used[op.idx]))
					m.addCons(name=f'fork_ub{op.idx}',
						cons=(quicksum(used[succ.idx] for succ in op.succ) <= 1))

				for succ in op.succ:
					m.addCons(name=f'dur{op.idx},{succ.idx}',
						cons=start[op.idx] + op.dur <= start[succ.idx] + 
							omega*(1-used[op.idx] + 1-used[succ.idx]))

		self.make_model_obj(m, self.inst.ops, start)

		return m, (used, start)
	
	def get_paths(self, used_val) -> List[List[Op]]:
		paths = []
		for train_ops in self.inst.ops:
			train_path = []

			op = train_ops[0]
			train_path.append(op)

			while op.n_succ > 0:
				op = train_path[-1]
				
				succ = [x for x in op.succ if used_val[x.idx] == 1]

				assert(len(succ) == 1)

				op = succ[0]
				train_path.append(op)

			paths.append(train_path)

		return paths
	

	def get_collisions(self, paths, start_val):
		res_uses = {}

		for path in paths:
			for op, succ in zip(path, path[1:] + [None]):
				for res in op.res:
					if not res.idx in res_uses:
						res_uses[res.idx] = []

					res_uses[res.idx].append(Res_use(
						op=op,
						succ=succ if succ else None,
						time=res.time,
						start=start_val[op.idx],
						end=start_val[succ.idx] + res.time if succ else None
					))

		collisions = []
		for res, uses in res_uses.items():
			for i, u1 in enumerate(uses):
				for u2 in uses[i+1:]:
					if u1.start < u2.end and u2.start < u1.end:
						collisions.append(Res_col(
							res=res,
							op1=u1.op,
							succ1=u1.succ,
							time1=u1.time,
							op2=u2.op,
							succ2=u2.succ,
							time2=u2.time
						))


		return collisions

	def make_collisions_model(self, paths: List[List[Op]], collisions: List[Res_col]):
		m = Model()
		m.setProbName('collisions')

		order = {}
		start = {}

		omega = self.inst.total_dur
		
		for path in paths:
			for op in path:
				start[op.idx] = m.addVar(name=f'start{op.idx}', vtype='C', 
										 lb=op.start_lb, ub=op.start_ub)
				
			for op, succ in zip(path[:-1], path[1:]):
				m.addCons(name=f'dur{op.idx},{succ.idx}',
					cons=start[op.idx] + op.dur <= start[succ.idx])
				
		for c in collisions:
			k1 = (c.res, c.op1.idx, c.op2.idx)
			k2 = (c.res, c.op2.idx, c.op1.idx)

			order[k1] = m.addVar(name=f'order{c.res},{c.op1.idx},{c.op2.idx}', vtype='B')
			order[k2] = m.addVar(name=f'order{c.res},{c.op2.idx},{c.op1.idx}', vtype='B')

			m.addCons(name=f'order{c.res},{c.op1.idx},{c.op2.idx}',
				cons=start[c.succ1.idx] + c.time1 <= start[c.op2.idx] + omega*(1-order[k1]))
			
			m.addCons(name=f'order{c.res},{c.op2.idx},{c.op1.idx}',
				cons=start[c.succ2.idx] + c.time2 <= start[c.op1.idx] + omega*(1-order[k2]))
			
			m.addCons(order[k1] + order[k2] == 1)

		self.make_model_obj(m, paths, start)

		return m, (order, start)
	
	def get_orders_from_collisions(self, collisions: List[Res_col], order_val):
		res_orders = []

		for c in collisions:
			k1 = (c.res, c.op1.idx, c.op2.idx)
			k2 = (c.res, c.op2.idx, c.op1.idx)

			assert(order_val[k1] == 1 or order_val[k2] == 1)
			if order_val[k1] == 1:
				res_orders.append(Res_order(
					first=c.op1,
					second=c.op2,
					time=c.time1
				))
			else:
				res_orders.append(Res_order(
					first=c.op2,
					second=c.op1,
					time=c.time2
				))

		return res_orders
		
	def add_orders_to_forks_model(self, m_f, used, start, orders: List[Res_order]):
		omega = self.inst.total_dur

		for o in orders:
			for succ in o.first.succ:
				m_f.addCons(name=f'order{succ.idx},{o.second}',
					cons=start[succ.idx] + o.time <= start[o.second.idx])
					#   + 
					# 	omega*(3 - used[o.first.idx] - used[o.second.idx] - used[succ.idx]))
			
	def add_orders_to_collisions_model(self, m_c, used_val, start, orders: List[Res_order]):
		omega = self.inst.total_dur

		for o in orders:
			for succ in o.first.succ:
				if used_val[o.first.idx] == 0 or used_val[o.second.idx] == 0:
					continue

				succ = [x for x in o.first.succ if used_val[x.idx] == 1][0]

				m_c.addCons(name=f'order{succ.idx},{o.second}',
					cons=start[succ.idx] + o.time <= start[o.second.idx])

	def solve(self):
		res_orders = []

		while True:
			m_f, (used, start) = self.make_forks_model()
			self.add_orders_to_forks_model(m_f, used, start, res_orders)

			m_f.writeProblem()
			# m_f.hideOutput()
			m_f.optimize()
			assert(m_f.getStatus() == 'optimal')
			print('forks', m_f.getObjVal())

			used_val = { k: int(round(m_f.getVal(v))) for k, v in used.items() }
			start_val = { k: int(round(m_f.getVal(v))) for k, v in start.items() }

			paths = self.get_paths(used_val)
			collisions = self.get_collisions(paths, start_val)
			if len(collisions) == 0:
				break

			m_c, (order, start_c) = self.make_collisions_model(paths, collisions)
			self.add_orders_to_collisions_model(m_c, used_val, start_c, res_orders)

			# m_c.hideOutput()
			m_c.optimize()
			assert(m_c.getStatus() == 'optimal')
			print('res', m_c.getObjVal())
			
			order_val = { k: int(round(m_c.getVal(v))) for k, v in order.items() }
			new_res_orders = self.get_orders_from_collisions(collisions, order_val)
			res_orders.extend(new_res_orders)

if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)
	sol = Solver(inst)
	sol.solve()

	# for op in sol.get_start_ops():
	#	 print(op)

	
	# for op in sol.get_end_ops():
	#	 print(op)