#!.venv/bin/python3

import sys

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, NamedTuple

from ortools.sat.python import cp_model as cp

from instance import Instance, Op, Op_idx


DEFAULT_DATA = 'data/nor1_critical_0.json'


@dataclass
class Event:
	idx: Op_idx
	start: int
	end: int

	def __str__(self):
		return f'Event({self.idx.train}, {self.idx.op}: {self.start} - {self.end})'


@dataclass
class Solution:
	events: Dict[int, List[Event]] = field(default_factory=lambda: defaultdict(list))


class Solver:
	inst: Instance
	curr_sol: Solution

	free: List[int]
	semi: List[int]
	fixed: List[int]

	max_dur: int
	res_used: Dict[int, List[int]]

	model: cp.CpModel
	solver: cp.CpSolver

	def __init__(self, inst: Instance, curr_sol: Solution = None, free: List[int] = [], semi: List[int] = [], fixed: List[int] = []):
		self.inst = inst
		self.curr_sol = curr_sol
		
		self.free = free
		self.semi = semi
		self.fixed = fixed

		self.max_dur = sum(inst.trains[t].max_dur for t in free + semi + fixed)

		self.model = cp.CpModel()
		self.solver = cp.CpSolver()

		self.create_res_used()

		self.create_op_vars()
		self.create_res_vars()
		self.add_path_cons()
		self.add_time_cons()
		self.add_res_interval_cons()


	def solve(self):
		self.solver.parameters.num_workers = 8

		status = self.solver.Solve(self.model)

		if status == cp.INFEASIBLE:
			return None
		
		return self.get_solution()


	def get_solution(self) -> Solution:
		sol = Solution()
		sol.events = {}

		for t in self.free:
			events = []

			train = self.inst.trains[t]

			for op in train.ops:
				if round(self.solver.value(self.var_op_used[op.idx])) == 1:
					start = round(self.solver.value(self.var_op_start[op.idx]))
					end = round(self.solver.value(self.var_op_end[op.idx]))
					events.append(Event(op.idx, start, end))
			
			sol.events[t] = events

		return sol

	def create_op_vars(self):
		self.var_op_used 	= {}
		self.var_op_start 	= {}
		self.var_op_end 	= {}
		self.var_op_path 	= {}

		for t in self.free:
			train = self.inst.trains[t]

			for op in train.ops:
				self.var_op_used[op.idx] 	= self.model.NewBoolVar(name=f'used_{t}_{op.i}')
				self.var_op_start[op.idx] 	= self.model.NewIntVar(name=f'start_{t}_{op.i}', lb=op.start_lb, ub=op.start_ub)
				self.var_op_end[op.idx] 	= self.model.NewIntVar(name=f'end_{t}_{op.i}', lb=op.start_lb + op.dur, ub=self.max_dur)

				for s in op.succ:
					self.var_op_path[t, op.i, s] = self.model.NewBoolVar(name=f'path_{t}_{op.i}_{s}')


	def create_res_used(self):
		self.res_used = defaultdict(set)

		for t in self.free:
			train = self.inst.trains[t]

			for op in train.ops:
				for r in op.res:
					self.res_used[r].add(t)

		print(self.res_used)


	def create_res_vars(self):
		self.var_res_lock 	= {}
		self.var_res_unlock = {}
		self.var_res_size 	= {}
		self.var_res_used	= {}

		for r, trains in self.res_used.items():
			for t in trains:
				if t in self.free:
					self.var_res_lock[r, t] 	= self.model.NewIntVar(lb=0, ub=self.max_dur, name=f'lock_{r}_{t}')
					self.var_res_unlock[r, t]	= self.model.NewIntVar(lb=0, ub=self.max_dur, name=f'unlock_{r}_{t}')
					self.var_res_size[r, t]		= self.model.NewIntVar(lb=0, ub=self.max_dur, name=f'size_{r}_{t}')

					if r in self.inst.trains[t].avoidable_res:
						self.var_res_used[r, t] = self.model.NewBoolVar(name=f'res_used_{r}_{t}')


	def add_path_cons(self):
		for t in self.free:
			train = self.inst.trains[t]

			for op in train.ops:
				
				if op.n_prev > 0:
					self.model.add(self.var_op_used[op.idx] == sum(self.var_op_path[t, p, op.i] for p in op.prev))
				else:
					self.model.add(self.var_op_used[op.idx] == 1)

				if op.n_succ > 0:
					self.model.add(self.var_op_used[op.idx] == sum(self.var_op_path[t, op.i, s] for s in op.succ))
			
			self.model.add(sum(self.var_op_used[op.idx] for op in train.ops if op.n_succ == 0) == 1)
	

	def add_time_cons(self):
		for t in self.free:
			train = self.inst.trains[t]

			for op in train.ops:
				if op.n_succ > 0:
					self.model.add(
						self.var_op_start[op.idx] + op.dur <= self.var_op_end[op.idx]
					).OnlyEnforceIf(self.var_op_used[op.idx])

					for s in op.succ:
						self.model.add(
							self.var_op_end[op.idx] == self.var_op_start[Op_idx(t, s)]
						).OnlyEnforceIf(self.var_op_path[t, op.i, s])
				else:
					self.model.add(
						self.var_op_start[op.idx] + op.dur == self.var_op_end[op.idx]
					)


	def add_res_interval_cons(self):

		for r, trains in self.res_used.items():
			if len(trains) == 1:
				continue

			interval_vars = []

			for t in trains:
				if t in self.free:
					train = self.inst.trains[t]

					for i in train.res_to_op[r]:
						op = train.ops[i]
						self.model.add(
							self.var_res_lock[r, t] <= self.var_op_start[op.idx]
						).OnlyEnforceIf(self.var_op_used[op.idx])
						
						self.model.add(
							self.var_res_unlock[r, t] >= self.var_op_end[op.idx] + self.inst.res_time(r, op.idx, 1)
						).OnlyEnforceIf(self.var_op_used[op.idx])
					
					if r in train.avoidable_res:
						interval_vars.append(self.model.NewOptionalIntervalVar(
							start		=self.var_res_lock[r, t], 
							size		=self.var_res_size[r, t], 
							end			=self.var_res_unlock[r, t], 
							is_present	=self.var_res_used[r, t],
							name		=f'interval_{r}_{t}'))
					
						self.model.add(
							sum(self.var_op_used[op.idx] for op in train.ops if r in op.res) == 0
						).OnlyEnforceIf(self.var_res_used[r, t].Not())

					else:
						interval_vars.append(self.model.NewIntervalVar(
							start	=self.var_res_lock[r, t], 
							size	=self.var_res_size[r, t], 
							end		=self.var_res_unlock[r, t],
							name	=f'interval_{r}_{t}'))
				
				
			self.model.AddNoOverlap(interval_vars)


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	solver = Solver(inst=inst, free=list(range(inst.n_trains)))
	
	sol = solver.solve()

	if sol:
		for k, v in sol.events.items():
			print(k, v)

	else:
		print('infeasible')