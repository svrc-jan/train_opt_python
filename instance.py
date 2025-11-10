#!.venv/bin/python3

import sys
import json

from collections import deque, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Dict, Set, NamedTuple

import networkx as nx

DEFAULT_DATA = 'data/nor1_critical_0.json'
MAX_DUR = 10000000


@dataclass
class Obj:
	threshold: int = 0
	coeff: int = 0
	increment: int = 0


@dataclass
class Op:
	idx: int = -1
	train: int = -1

	dur: int = 0
	start_lb: int = 0
	start_ub: int = MAX_DUR

	succ: List[int] = field(default_factory=list)
	prev: List[int] = field(default_factory=list)

	res: List[int] = field(default_factory=list)

	obj: Obj|None = None


	@cached_property
	def n_succ(self):
		return len(self.succ)


	@cached_property
	def n_prev(self):
		return len(self.prev)


	@cached_property
	def n_res(self):
		return len(self.res)
	
	def __str__(self):
		return f'Op({self.idx.train}, {self.idx.op})'


@dataclass
class Train:
	idx: int = -1

	op_start: int = -1
	op_end: int = -1 
	ops: List[Op] = field(default_factory=list)
	
	max_dur: int = MAX_DUR
	res_to_op: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
	avoidable_res: Set[int] = field(default_factory=set)
	mandatory_res: Set[int] = field(default_factory=set)


	@property
	def op_last(self):
		return self.op_end - 1


	@property
	def n_ops(self):
		return len(self.ops)


	@property
	def res(self):
		return self.avoidable_res | self.mandatory_res


class Instance:
	ops: List[Op]
	trains: List[Train]
	points_valid: bool
	max_dur: int
	res_occur: List[int]
	res_to_train: List[Set[int]]

	__res_time: List[Dict[int, int]]
	__res_name_idx: Dict[str, int]

	def __init__(self, jsn_file: str):
		self.parse_json_file(jsn_file)
		self.make_prev_ops()
		self.make_max_dur()
		self.make_avoidable_resources()
		self.make_res_occur()


	def parse_json_file(self, jsn_file: str):
		self.ops = []
		self.trains = []
		self.__res_time = []
		self.__res_name_idx = {}

		with open(jsn_file, 'r') as fd:
			jsn = json.load(fd)
		
		for jsn_train in jsn['trains']:
			self.parse_json_train(jsn_train)

		for jsn_obj in jsn['objective']:
			self.parse_json_obj(jsn_obj)

		for train in self.trains:
			train.ops = self.ops[train.op_start:train.op_end]

	def parse_json_train(self, jsn_train: dict):
		train = Train(idx=self.n_trains, op_start=self.n_ops)
		
		for jsn_op in jsn_train:
			self.parse_json_op(jsn_op, train)

		train.op_end = self.n_ops
		self.trains.append(train)

		# check if only last op is ending op (n_succ == 0), required for solver
		for op in self.ops[train.op_start:train.op_last]:
			assert(op.n_succ > 0)

		assert(self.ops[train.op_last].n_succ == 0)


	def parse_json_op(self, jsn_op: dict, train: Train):
		op = Op(idx=self.n_ops, train=train.idx)

		op.dur = jsn_op['min_duration']
		op.start_lb = jsn_op.get('start_lb', 0)
		op.start_ub = jsn_op.get('start_ub', MAX_DUR)

		op.succ = [x + train.op_start for x in jsn_op['successors']]

		for jsn_res in jsn_op.get('resources', []):
			res_idx = self.get_res_idx(jsn_res['resource'])
			res_time = jsn_res.get('release_time', 0)
			
			op.res.append(res_idx)
			train.res_to_op[res_idx].append(op.idx)

			if res_time > 0:
				self.__res_time[res_idx][op.idx] = res_time


		self.ops.append(op)


	def parse_json_obj(self, jsn_obj):
		if jsn_obj['type'] != 'op_delay':
			return
		
		obj = Obj(
			threshold	=jsn_obj.get('threshold', 0),
			coeff		=jsn_obj.get('coeff', 0),
			increment	=jsn_obj.get('increment', 0)
		)

		self.ops[self.trains[jsn_obj['train']].op_start + jsn_obj['operation']].obj = obj


	def make_prev_ops(self):
		for op in self.ops:
			for succ in op.succ:
				self.ops[succ].prev.append(op.idx)


	def get_res_idx(self, name):
		idx = self.__res_name_idx.get(name, -1)
		
		if idx == -1:
			idx = len(self.__res_name_idx)
			self.__res_name_idx[name] = idx
			self.__res_time.append({})

		return idx


	def make_max_dur(self):
		for train in self.trains:
			self.make_train_max_dur(train)
		
		self.max_dur = sum(t.max_dur for t in self.trains)

		for train in self.trains:
			for op in train.ops:
				if op.start_ub == MAX_DUR:
					op.start_ub = self.max_dur


	def make_train_max_dur(self, train: Train):
		in_ord = { op.idx : op.n_prev for op in train.ops}
		start = { op.idx : op.start_lb for op in train.ops}

		q = deque([train.op_start])

		while q:
			idx = q.popleft()
			op = self.ops[idx]

			for succ in op.succ:
				in_ord[succ] -= 1

				if in_ord[succ] == 0:
					q.append(succ)

		train.max_dur = start[train.op_last] + self.ops[train.op_last].dur 


	def make_avoidable_resources(self):
		for train in self.trains:
			for r in train.res_to_op.keys():
				if self.is_resource_avoidable(train, r):
					train.avoidable_res.add(r)
				else:
					train.mandatory_res.add(r)


	def is_resource_avoidable(self, train: Train, res: int):
		if res in train.ops[0].res or res in train.ops[-1].res:
			return False

		g = nx.DiGraph()
		g.add_nodes_from(range(train.op_start, train.op_end))

		for op in train.ops:
			g.add_edges_from((op.idx, s) for s in op.succ)

		for op in train.ops:
			if res in op.res:
				g.remove_node(op.idx)

		return nx.has_path(g, train.op_start, train.op_last)	


	def make_res_occur(self):
		self.res_occur = [0] * self.n_res
		self.res_to_train = [set()] * self.n_res

		for train in self.trains:
			for r in train.res:
				self.res_to_train[r].add(train.idx)

		for op in self.ops:
			for r in op.res:
				self.res_occur[r] += 1


	def res_time(self, res: int, op: int, min_time: int = 0) -> int:
		return min(self.__res_time[res].get(op, 0), min_time)


	@property
	def n_ops(self):
		return len(self.ops)

	@property
	def n_trains(self):
		return len(self.trains)
	
	@property
	def n_res(self):
		return len(self.__res_name_idx)

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	
	print(inst.res_to_train)
	# for train in inst.trains:
	# 	print(train.idx, len(train.avoidable_res), len(train.mandatory_res))
