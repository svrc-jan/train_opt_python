#!.venv/bin/python3

import sys

from collections import defaultdict
import gurobipy as gp

from gurobipy import GRB

from instance import Instance


MAX_DUR = 100000
DEFAULT_DATA = 'data/nor1_critical_0.json'

class Model:
	inst: Instance
	gm: gp.Model

	def __init__(self, inst):
		self.inst = inst
		self.gm = gp.Model()

	
	def build(self):
		self.add_var_level_time()
		self.add_var_op_used()
		self.add_var_res()
		self.add_var_obj()
		
		self.add_cons_dur()
		self.add_cons_flow()
		self.add_cons_res_interval()
		self.add_cons_obj()


	def add_var_level_time(self):
		self.var_level_time = {}

		for level in self.inst.levels:
			self.var_level_time[level.idx] = self.gm.addVar(
				lb=level.time_lb, ub=level.time_ub,
				vtype=GRB.CONTINUOUS, name=f'level_time_{level.idx}')


	def add_var_op_used(self):
		self.var_op_used = {}

		for op in self.inst.ops:
			self.var_op_used[op.idx] = self.gm.addVar(
				vtype=GRB.BINARY, name=f'op_used_{op.idx}')

	
	def add_var_res(self):
		self.var_res_lock = {}
		self.var_res_unlock = {}
		self.var_res_order = {}

		for train in self.inst.trains:
			for r in train.res:
				self.var_res_lock[r, train.idx] = self.gm.addVar(
					lb=0, ub=MAX_DUR, vtype=GRB.CONTINUOUS, name=f'res_lock_{r}_{train.idx}')
				
				self.var_res_unlock[r, train.idx] = self.gm.addVar(
					lb=0, ub=MAX_DUR, vtype=GRB.CONTINUOUS, name=f'res_unlock_{r}_{train.idx}')
				
	
	def add_cons_dur(self):
		level_time = self.var_level_time
		op_used = self.var_op_used

		for op in self.inst.ops:
			self.gm.addConstr(level_time[op.level_start] + op.dur*op_used[op.idx] <= level_time[op.level_end])
		

	def add_cons_flow(self):
		op_used = self.var_op_used

		for level in self.inst.levels:
			if level.n_ops_in == 0:
				self.gm.addConstr(sum(op_used[op_out] for op_out in level.ops_out) == 1)

			elif level.n_ops_out == 0:
				self.gm.addConstr(sum(op_used[op_in] for op_in in level.ops_in) == 1)

			else:
				self.gm.addConstr(
					sum(op_used[op_out] for op_out in level.ops_out) == 
					sum(op_used[op_in] for op_in in level.ops_in))

	
	def add_cons_res_interval(self, min_res_time=1):
		M = MAX_DUR

		time = self.var_level_time
		used = self.var_op_used
		lock = self.var_res_lock
		unlock = self.var_res_unlock

		for op in self.inst.ops:
			for res in op.res:
				self.gm.addConstr(lock[res.idx, op.train] <= time[op.level_start] + M*(1 - used[op.idx]))
				self.gm.addConstr(time[op.level_end] + max(res.time, min_res_time)
					<= unlock[res.idx, op.train] +  M*(1 - used[op.idx]))


	def add_var_obj(self):
		self.var_obj = {}

		for op in self.inst.ops:
			if op.obj:
				if op.obj.is_bin:
					self.var_obj[op.idx] = self.gm.addVar(vtype=GRB.BINARY, name=f'obj_bin_{op.idx}')
				else:
					self.var_obj[op.idx] = self.gm.addVar(lb=0, ub=MAX_DUR,	vtype=GRB.CONTINUOUS, name=f'obj_{op.idx}')
					
		
	def add_cons_obj(self):
		M = MAX_DUR

		time = self.var_level_time
		used = self.var_op_used

		for op in self.inst.ops:
			if op.obj:
				if op.obj.is_bin:
					self.gm.addConstr(time[op.level_start] - op.obj.time <= 
						M*self.var_obj[op.idx] + M*(1 - used[op.idx]))
				else:
					self.gm.addConstr(time[op.level_start] - op.obj.time <= 
					   	self.var_obj[op.idx] + M*(1 - used[op.idx]))

	
	def set_inst_obj(self):
		self.gm.setObjective(sum(self.var_obj[op.idx]*op.obj.value for op in self.inst.ops if not op.obj is None))


	def add_cons_res_overlap(self, res, train1, train2):
		M = MAX_DUR

		order = self.var_res_order
		lock = self.var_res_lock
		unlock = self.var_res_unlock

		if train1 > train2:
			train1, train2 = train2, train1

		k = (res, train1, train2)

		if k in order:
			print(f'cond1: {unlock[res, train1].X:.3f} <= {lock[res, train2].X:.3f} + {M*(1 - order[k].X):.3f}')
			print(f'cond2: {lock[res, train1].X:.3f} <= {unlock[res, train2].X:.3f} + {M*order[k].X:.3f}')

			exit(-1)

		order[k] = self.gm.addVar(vtype=GRB.BINARY, name=f'res_order_{res}_{train1}_{train2}')
		
		self.gm.addConstr(unlock[res, train1] <= lock[res, train2] + M*(1 - order[k]))
		self.gm.addConstr(unlock[res, train2] <= lock[res, train1] + M*order[k])

		
	def get_result_res_uses(self):
		used = self.var_op_used
		time = self.var_level_time

		res_uses = defaultdict(dict)
		for op in self.inst.ops:
			if used[op.idx].X > 0.5:
				op_start = time[op.level_start].X
				op_end = time[op.level_end].X

				for res in op.res:
					op_release = op_end + res.time
					if op.train in res_uses[res.idx]:
						res_start, res_end = res_uses[res.idx][op.train]
						res_uses[res.idx][op.train] = (min(op_start, res_start), max(op_release, res_end))
					else:
						res_uses[res.idx][op.train] = (op_start, op_release)


		res_uses = { res: [(t1, t2, train) for train, (t1, t2) in ru.items()] for (res, ru) in res_uses.items() }

		
		for ru in res_uses.values():
			ru.sort()
		
		return res_uses


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	model = Model(inst)
	model.build()
	model.set_inst_obj()
	model.gm.write('model.mps')
	model.gm.optimize()
	print(model.get_result_res_uses())
