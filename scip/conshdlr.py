#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use

class Res_conshdlr(scip.Conshdlr):
	def __init__(self, model, res_uses, max_train_dur):
		super().__init__()
		self.model = model
		self.res_uses = res_uses
		self.max_train_dur = max_train_dur

	def get_res_collisions(self, solution):
		eps = 1e-5

		model = self.model

		s = model.data['s']
		e = model.data['e']
		f = model.data['f']

		collisions = []

		used = {}

		for res_idx, res_uses in self.res_uses.items():
			for i, ru1 in enumerate(res_uses):
				op1 = ru1.op
				if not op1 in used:
					used[op1] = (op1.n_succ == 0) or \
						(sum(model.getSolVal(solution, f[op1, succ1]) for succ1 in op1.succ) >= 1 - eps)
				
				if not used[op1]:
					continue

				s1 = int(round(model.getSolVal(solution, s[op1])))
				e1 = int(round(model.getSolVal(solution, e[op1]) + ru1.time))

				for ru2 in res_uses[i+1:]:
					op2 = ru2.op

					if op1.train_idx == op2.train_idx:
						continue

					if not op2 in used:
						used[op2] = (op2.n_succ == 0) or \
							(sum(model.getSolVal(solution, f[op2, succ2]) for succ2 in op2.succ) >= 1 - eps)
				
					if not used[op2]:
						continue

					s2 = int(round(model.getSolVal(solution, s[op2])))
					e2 = int(round(model.getSolVal(solution, e[op2]) + ru2.time))

					if (s1 < e2) and (s2 < e1):
						collisions.append((res_idx, ru1, ru2))

		return collisions

	# s1 <= smax
 	# s2 <= smax
	# 
	# emin <= e1
	# emin <= e2
	# 
	# emin <= smax

	def make_res_cons(self, collision):
		model = self.model

		
		s = model.data['s']
		e = model.data['e']
		f = model.data['f']
		r = model.data['r']

		res_idx, ru1, ru2 = collision

		op1 = ru1.op
		op2 = ru2.op

		use_op1 = scip.quicksum(f[op1, succ1] for succ1 in op1.succ) if op1.n_succ > 0 else 1
		use_op2 = scip.quicksum(f[op2, succ2] for succ2 in op2.succ) if op2.n_succ > 0 else 1

		M1 = max(self.max_train_dur[op1.train_idx],	self.max_train_dur[op2.train_idx])

		k1 = (res_idx, op1, op2)
		k2 = (res_idx, op2, op1)

		if not k1 in r:
			r[k1] = model.addVar(name=f'r{res_idx},{op1},{op2}', vtype='B')
		
		v1 = r[k1]
		
		if not k2 in r:
			r[k2] = model.addVar(name=f'r{res_idx},{op2},{op1}', vtype='B')

		v2 = r[k2]

		cons = v1 + v2 == 1
		model.addCons(name=f'ch{res_idx},{op1},{op2}', cons=cons, 
			removable=True)		

		cons1 = e[op1] + ru1.time <= s[op2] + M1*(1 - v1)
		cons2 = e[op2] + ru2.time <= s[op1] + M1*(1 - v2)
		
		model.addCons(name=f'res{res_idx},{op1},{op2}', cons=cons1,
			modifiable=True, removable=True)
		model.addCons(name=f'res{res_idx},{op2},{op1}', cons=cons2, 
			modifiable=True, removable=True)

	def make_conss_from_collisions(self, collisions):
		model = self.model
		if not 'r' in model.data:
			model.data['r'] = {}

		for col in collisions:
			self.make_res_cons(col)
			

	def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
		collisions = self.get_res_collisions(solution)
	
		if len(collisions) == 0:
			return {"result": scip.SCIP_RESULT.FEASIBLE}
		
		return {"result": scip.SCIP_RESULT.INFEASIBLE}

	def consenfolp(self, constraints, nusefulconss, solinfeasible):
		collisions = self.get_res_collisions(solution=None)
		# print(nusefulconss, constraints)
		print('enf lp')
		if len(collisions) == 0:
			return {"result": scip.SCIP_RESULT.FEASIBLE}
		
		self.make_conss_from_collisions(collisions)
		return {"result": scip.SCIP_RESULT.CONSADDED}
	
	def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
		collisions = self.get_res_collisions(solution=None)
		print('enf pseudo')
		if len(collisions) == 0:
			return {"result": scip.SCIP_RESULT.FEASIBLE}
		
		self.make_conss_from_collisions(collisions)
		return {"result": scip.SCIP_RESULT.CONSADDED}


	def conslock(self, constraint, locktype, nlockspos, nlocksneg):
		pass