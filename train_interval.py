#!.venv/bin/python3

import sys


import gurobipy as gp

from gurobipy import GRB

from instance import Instance, Op


DEFAULT_DATA = 'data/nor1_critical_0.json'

class Model:
	inst: Instance
	gm: gp.Model

	def __init__(self, inst):
		self.inst = inst
		self.gm = gp.Model()

	
	def build(self):
		self.add_op_vars()
		self.add_res_vars()
		self.add_obj_var()

		self.add_dur_cons()
		self.add_flow_cons()
		self.add_res_interval_cons()
		self.add_threshold_cons()

	
	def add_op_vars(self):
		self.var_op_start = {}
		self.var_op_flow = {}

		for op in self.inst.ops:
			assert(op.start_lb <= op.start_ub)

			self.var_op_start[op.idx] = self.gm.addVar(
				lb=op.start_lb, ub=op.start_ub, vtype=GRB.CONTINUOUS, name=f'op_start_{op.idx}')
			
			for succ in op.succ:
				self.var_op_flow[op.idx, succ] = self.gm.addVar(
					vtype=GRB.BINARY, name=f'op_flow_{op.idx}_{succ}')

	
	def add_res_vars(self):
		self.var_res_lock = {}
		self.var_res_unlock = {}
		self.var_res_order = {}

		for train in self.inst.trains:
			for r in train.res:
				self.var_res_lock[r, train.idx] = self.gm.addVar(
					lb=0, ub=inst.max_dur, vtype=GRB.CONTINUOUS, name=f'res_lock_{r}_{train.idx}')
				
				self.var_res_unlock[r, train.idx] = self.gm.addVar(
					lb=0, ub=inst.max_dur, vtype=GRB.CONTINUOUS, name=f'res_lock_{r}_{train.idx}')
				
	
	def add_dur_cons(self):
		M = self.inst.max_dur

		start = self.var_op_start
		flow = self.var_op_flow

		for op in self.inst.ops:
			if op.n_succ == 0:
				pass
			elif op.n_succ == 0:
				self.gm.addConstr(start[op.idx] + op.dur <= start[op.succ[0]])
			else:
				for succ in op.succ:
					self.gm.addConstr(start[op.idx] + op.dur <= start[succ] + M*(1 - flow[op.idx, succ]))
		

	def add_flow_cons(self):
		flow = self.var_op_flow

		for op in self.inst.ops:
			if op.n_prev == 0:
				self.gm.addConstr(sum(flow[op.idx, succ] for succ in op.succ) == 1)

			elif op.n_succ == 0:
				self.gm.addConstr(sum(flow[prev, op.idx] for prev in op.prev) == 1)

			else:
				self.gm.addConstr(
					sum(flow[op.idx, succ] for succ in op.succ) == 
					sum(flow[prev, op.idx] for prev in op.prev))

	
	def add_res_interval_cons(self):
		M = self.inst.max_dur

		start = self.var_op_start
		flow = self.var_op_flow
		lock = self.var_res_lock
		unlock = self.var_res_unlock

		for train in self.inst.trains:
			for r in train.res:
				for idx in train.res_to_op[r]:
					op = self.inst.ops[idx]

					if op.n_succ > 0:
						self.gm.addConstr(lock[r, train.idx] <= start[idx] + M*(1 - sum(flow[idx, succ] for succ in op.succ)))
						for succ in op.succ:
							self.gm.addConstr(unlock[r, train.idx] >= start[succ] + M*(1 - flow[idx, succ]))

					else:
						self.gm.addConstr(lock[r, train.idx] <= start[idx])
						self.gm.addConstr(unlock[r, train.idx] >= start[idx] + op.dur)


	def add_obj_var(self):
		self.var_obj = {}

		for op in self.inst.ops:
			if op.obj:
				if op.obj.coeff > 0:
					self.var_obj[op.idx] = self.gm.addVar(
						lb=0, ub=self.inst.max_dur, obj=op.obj.coeff, vtype=GRB.CONTINUOUS, name=f'obj_{op.idx}')
				elif op.obj.threshold > 0:
					self.var_obj[op.idx] = self.gm.addVar(
						obj=op.obj.threshold, vtype=GRB.BINARY, name=f'obj_bin_{op.idx}')
					
		
	def add_threshold_cons(self):
		M = self.inst.max_dur

		start = self.var_op_start

		for op in self.inst.ops:
			if op.obj:
				if op.obj.coeff > 0:
					self.gm.addConstr(start[op.idx] - op.obj.threshold <= self.var_obj[op.idx])

				elif op.obj.threshold > 0:
					self.gm.addConstr(start[op.idx] - op.obj.threshold <= M*self.var_obj[op.idx])


	def g

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	model = Model(inst)
	model.build()
	model.gm.write('model.mps')
	model.gm.optimize()
