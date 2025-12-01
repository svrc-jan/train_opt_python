#!.venv/bin/python3

import sys
import json


from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Dict, Set

from disjoint_set import Disjoint_set
from base_inst import Base_inst


DEFAULT_DATA = 'data/smi_headway_5.json'
MAX_DUR = 100000

@dataclass
class Res:
	idx: int = -1
	time: int = 0


@dataclass
class Obj:
	time: int = 0
	value: int = 0
	is_bin: bool = False


@dataclass
class Op:
	idx: int = -1
	train: int = -1

	level_start: int = -1
	level_end: int = -1

	dur: int = 0
	start_lb: int = 0
	start_ub: int = MAX_DUR

	res: List[Res] = field(default_factory=list)

	obj: Obj|None = None


	@property
	def n_res(self) -> int:
		return len(self.res)
	
	@property
	def has_obj(self) -> int:
		return not self.obj is None

	def __str__(self) -> int:
		return f'Op({self.idx})'


@dataclass
class Level:
	idx: int = -1
	train: int = -1

	time_lb: int = 0
	time_ub: int = MAX_DUR

	ops_in: List[int] = field(default_factory=list)
	ops_out: List[int] = field(default_factory=list)


	@property
	def n_ops_in(self) -> int:
		return len(self.ops_in)
	
	@property
	def n_ops_out(self) -> int:
		return len(self.ops_out)


@dataclass
class Train:
	idx: int = -1

	op_start: int = -1
	op_end: int = -1

	level_start: int = -1
	level_end: int = -1

	res: Set[int] = field(default_factory=set)


	@property
	def op_last(self) -> int:
		return self.op_end - 1

	@property
	def level_last(self) -> int:
		return self.level_end - 1

	@property
	def n_ops(self) -> int:
		return len(self.op_end - self.op_start)


class Instance:
	base_inst: Base_inst

	trains: List[Train]
	levels: List[Level]
	ops: List[Op]
	
	__res_name_idx: Dict[str, int]

	def __init__(self, jsn_file: str):
		self.base_inst = Base_inst(jsn_file)
		self.add_trains_ops()
		self.add_levels()


	def add_trains_ops(self):
		self.trains = []
		self.ops = []

		self.__res_name_idx = {}
		self.__res_time = []

		for base_train in self.base_inst.trains:
			train = Train(idx=self.n_trains)

			train.op_start = self.n_ops

			for base_op in base_train.ops:
				op = Op(
					idx		=self.n_ops, 
					train	=train.idx,
					dur		=base_op.dur,
					start_lb=base_op.start_lb,
					start_ub=base_op.start_ub
				)

				if op.start_ub == -1:
					op.start_ub = MAX_DUR

				for base_res in base_op.res:
					res = Res(idx=self.res_idx(base_res.name), time=base_res.time)
					op.res.append(res)

					train.res.add(res.idx)
				
				self.ops.append(op)
			
			train.op_end = self.n_ops
			self.trains.append(train)


		for base_obj in self.base_inst.objs:
			is_bin = base_obj.increment > 0

			obj = Obj(
				time	=base_obj.threshold,
				value	=base_obj.increment if is_bin else base_obj.coeff,
				is_bin	=is_bin
			)

			self.ops[base_obj.op + self.trains[base_obj.train].op_start].obj = obj


	def add_levels(self):
		self.levels = []

		for train in self.trains:
			base_train = self.base_inst.trains[train.idx]

			disj_set = Disjoint_set(base_train.n_ops)

			for base_op in base_train.ops:
				for i, s1 in enumerate(base_op.succ):
					for s2 in base_op.succ[i+1:]:
						disj_set.union_set(s1, s2)
			
			train.level_start = self.n_levels

			for succ_set in disj_set.get_sets():
				level = Level(idx=self.n_levels, train=train.idx)

				for o in succ_set:
					self.ops[o + train.op_start].level_start = level.idx

				self.levels.append(level)

			last_level = Level(idx=self.n_levels, train=train.idx)
			self.levels.append(last_level)
			
			train.level_start = self.n_levels
		
			for o, base_op in enumerate(base_train.ops):
				if base_op.n_succ == 0:
					self.ops[o + train.op_start].level_end = last_level.idx
				else:
					self.ops[o + train.op_start].level_end = self.ops[base_op.succ[0] + train.op_start].level_start

			
			for o, base_op in enumerate(base_train.ops):
				for s in base_op.succ:
					assert(self.ops[o + train.op_start].level_end == self.ops[s + train.op_start].level_start)


		for op in self.ops:
			assert(op.level_start != -1 and op.level_end != -1)
			self.levels[op.level_end].ops_in.append(op.idx)
			self.levels[op.level_start].ops_out.append(op.idx)

		
		for level in self.levels:
			level.time_lb = min((self.ops[o].start_lb for o in level.ops_out), default=0)
			level.time_ub = max((self.ops[o].start_ub for o in level.ops_out), default=MAX_DUR)


	def res_idx(self, name: str) -> int:
		idx = self.__res_name_idx.get(name, -1)
		
		if idx == -1:
			idx = len(self.__res_name_idx)
			self.__res_name_idx[name] = idx

		return idx


	@property
	def n_trains(self):
		return len(self.trains)


	@property
	def n_levels(self):
		return len(self.levels)


	@property
	def n_ops(self):
		return len(self.ops)

	
	


	@property
	def n_res(self):
		return len(self.__res_name_idx)

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
