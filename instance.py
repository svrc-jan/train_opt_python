#!.venv/bin/python3

import json
import sys
import itertools

from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import List, Tuple, Dict
from functools import cached_property

DEFAULT_DATA = 'data/testing/headway1.json'

@dataclass
class Res:
	idx: int
	time: int


@dataclass
class Obj:
	threshold: int	= 0
	coeff: int		= 0
	increment: int	= 0


@dataclass
class Op:
	train_idx: int	= -1
	op_idx: int		= -1

	dur: int		= 0
	start_lb: int	= 0
	start_ub: int|None	= None

	succ: List['Op']	= field(default_factory=list)
	prev: List['Op']	= field(default_factory=list)
	res: List[Res]		= field(default_factory=list)

	obj: Obj|None		= None

	@cached_property
	def n_succ(self) -> int:
		return len(self.succ)
	
	@cached_property
	def n_prev(self) -> int:
		return len(self.prev)

	@cached_property
	def n_res(self) -> int:
		return len(self.res)

	@cached_property
	def idx(self) -> int:
		return (self.train_idx, self.op_idx)

	def __hash__(self) -> int:
		return hash(self.idx)

	def __repr__(self) -> str:
		return f'Op({self.train_idx}, {self.op_idx})'


@dataclass
class Res_use:
	op: Op
	time: int


class Instance:
	ops: List[List[Op]]
	res_idx: Dict[str, int]
	res_uses: Dict[int, List[Res_use]]
	n_ops: List[int]
	
	max_dur: int
	max_train_dur: List[int]

	n_ops: List[int]
	n_train_ops: List[int]


	def __init__(self, json_file: str):
		self.parse_json(json_file)
		self.calc_max_train_dur()

	def parse_json(self, json_file: str) -> None:
		with open(json_file, 'r') as fd:
			jsn = json.load(fd)


		self.ops = []
		self.res_idx = {}
		self.res_uses = {}
		self.total_dur = 0

		for train_idx, train_jsn in enumerate(jsn['trains']):
			train_ops = []

			succ_l = []

			for op_idx, op_jsn in enumerate(train_jsn):
				op = Op(
					train_idx	=train_idx,
					op_idx		=op_idx,
					dur			=op_jsn['min_duration'],
					start_lb	=op_jsn.get('start_lb', 0),
					start_ub	=op_jsn.get('start_ub', None)
				)

				op.start_obj = op.start_ub
				self.total_dur += op.dur

				succ_l.append(op_jsn['successors'])

				for res_jsn in op_jsn.get('resources', []):
					res_name = res_jsn['resource']
					res_time = res_jsn.get('release_time', 0)
					
					if not res_name in self.res_idx:
						res_idx = len(self.res_idx)
						self.res_idx[res_name] = res_idx
					else:
						res_idx = self.res_idx[res_name]

					op.res.append(Res(res_idx, res_time))
					
					if not res_idx in self.res_uses:
						self.res_uses[res_idx] = []

					self.res_uses[res_idx].append(Res_use(op=op, time=res_time))

				train_ops.append(op)

			for op, succ_idx in zip(train_ops, succ_l):
				for succ in [train_ops[i] for i in succ_idx]:
					op.succ.append(succ)
					succ.prev.append(op)

			self.ops.append(train_ops)

		self.n_train_ops = [len(train_ops) for train_ops in self.ops]

		for obj_jsn in jsn['objective']:
			if obj_jsn['type'] != 'op_delay':
				continue

			self.ops[obj_jsn['train']][obj_jsn['operation']].obj = Obj(
				threshold	=obj_jsn.get('threshold', 0),
				coeff		=obj_jsn.get('coeff', 0),
				increment	=obj_jsn.get('increment', 0)
			)

	def calc_max_train_dur(self):
		dur = {}

		def rec(op: Op):
			if not op in dur:
				if op.n_succ == 0:
					dur[op] = op.dur
				
				else:
					dur[op] = op.dur + max(rec(succ) for succ in op.succ)
			
			return dur[op]

		self.max_train_dur = [rec(self.ops[tr][0]) for tr in range(self.n_trains)]
		
	@cached_property
	def n_trains(self) -> int:
		return len(self.ops)
	
	@cached_property
	def n_ops(self) -> int:
		return sum(self.n_train_ops)
	
	@cached_property
	def max_dur(self) -> int:
		return sum(self.max_train_dur)

	@cached_property
	def n_res(self) -> int:
		return len(self.res_idx)
	
	@cached_property
	def avg_dur(self) -> float:
		return self.total_dur/sum(self.n_ops)
	
	@property
	def all_ops(self):
		return itertools.chain(*self.ops)

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	print(inst.max_train_dur)

	
		