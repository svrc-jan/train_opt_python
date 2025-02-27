#!.venv/bin/python3

import json
import sys

from itertools import count
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

	start_obj: int|None = None

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

class Instance:
	ops: List[List[Op]]
	res_idx: Dict[str, int]
	res_ops: Dict[str, List[Op]]
	n_ops: List[int]
	total_dur: int

	def __init__(self, json_file: str):
		self.parse_json(json_file)
		self.calculate_dist()

	def parse_json(self, json_file: str) -> None:
		with open(json_file, 'r') as fd:
			jsn = json.load(fd)


		self.ops = []
		self.res_idx = {}
		self.res_ops = {}
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
					
					if not res_idx in self.res_ops:
						self.res_ops[res_idx] = []

					self.res_ops[res_idx].append(op)

				train_ops.append(op)

			for op, succ_idx in zip(train_ops, succ_l):
				for succ in [train_ops[i] for i in succ_idx]:
					op.succ.append(succ)
					succ.prev.append(op)

			self.ops.append(train_ops)

		self.n_ops = [len(train_ops) for train_ops in self.ops]

		for obj_jsn in jsn['objective']:
			if obj_jsn['type'] != 'op_delay':
				continue

			self.ops[obj_jsn['train']][obj_jsn['operation']].obj = Obj(
				threshold	=obj_jsn.get('threshold', 0),
				coeff		=obj_jsn.get('coeff', 0),
				increment	=obj_jsn.get('increment', 0)
			)

	def calculate_dist(self, train_idx=None):
		if train_idx is None:
			for idx in range(self.n_trains):
				self.calculate_dist(train_idx=idx)
			return

		def djikstra(start_op: Op, backward=True):
			
			q = PriorityQueue()
			cnt = count()
			q.put((0, next(cnt), start_op))

			dist = { start_op.op_idx: 0 }

			while not q.empty():
				d, _, op = q.get()
				if d > dist.get(op.op_idx, float('inf')):
					continue

				for n in (op.prev if backward else op.succ):
					n_d = d + (n.dur if backward else op.dur)
					if n_d < dist.get(n.op_idx, float('inf')):
						dist[n.op_idx] = n_d
						q.put((n_d, next(cnt), n))
			
			return dist
		
		for op in self.ops[train_idx]:
			if op.obj is not None:
				dist = djikstra(op)
				for k, v in dist.items():
					curr = self.ops[train_idx][k]
					curr.start_obj = op.obj.threshold - v if curr.start_obj is None else \
						min(curr.start_obj, op.obj.threshold - v)

		
		for op in self.ops[train_idx]:
			if op.obj is not None:
				dist = djikstra(op, backward=False)
				for k, v in dist.items():
					curr = self.ops[train_idx][k]
					if curr.start_obj is None:
						curr.start_obj = op.obj.threshold + v

			
		
	@cached_property
	def n_trains(self) -> int:
		return len(self.ops)

	@cached_property
	def n_res(self) -> int:
		return len(self.res_idx)

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	for tr in inst.ops:
		for op in tr:
			print(op, op.start_obj)

	
		