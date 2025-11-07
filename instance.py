#!.venv/bin/python3

import sys
import json

from collections import deque, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Dict, NamedTuple

import networkx as nx

DEFAULT_DATA = 'data/nor1_critical_0.json'
MAX_DUR = 10000000


@dataclass
class Obj:
	threshold: int = 0
	coeff: int = 0
	increment: int = 0

class Op_idx(NamedTuple):
	train: int = -1
	op: int = -1


@dataclass
class Op:
	idx: Op_idx = Op_idx()

	dur: int = 0
	start_lb: int = 0
	start_ub: int = MAX_DUR

	succ: List[int] = field(default_factory=list)
	prev: List[int] = field(default_factory=list)

	res: List[int] = field(default_factory=list)

	obj: Obj|None = None

	@property
	def i(self):
		return self.idx.op


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
	ops: List[Op] = field(default_factory=list)
	max_dur: int = MAX_DUR
	res_to_op: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
	avoidable_res: List[int] = field(default_factory=list)

	@cached_property
	def n_ops(self):
		return len(self.ops)


	@property
	def res(self):
		return self.res_to_op.keys()

class Instance:
	trains: List[Train]
	points_valid: bool
	max_dur: int

	__res_time: List[Dict[Op_idx, int]]
	__res_name_idx: Dict[str, int]

	def __init__(self, jsn_file: str):
		self.parse_json_file(jsn_file)
		self.make_prev_ops()
		self.calculate_max_dur()
		self.calculate_avoidable_resources()


	def parse_json_file(self, jsn_file: str):
		self.trains = []
		self.__res_time = []
		self.__res_name_idx = {}

		with open(jsn_file, 'r') as fd:
			jsn = json.load(fd)
		
		for jsn_train in jsn['trains']:
			self.parse_json_train(jsn_train)

		for jsn_obj in jsn['objective']:
			self.parse_json_obj(jsn_obj)


	def parse_json_train(self, jsn_train: dict):
		train = Train(idx=len(self.trains))
		
		for jsn_op in jsn_train:
			self.parse_json_op(jsn_op, train)

		self.trains.append(train)


	def parse_json_op(self, jsn_op: dict, train: Train):
		op = Op(idx=Op_idx(train=train.idx, op=len(train.ops)))

		op.dur = jsn_op['min_duration']
		op.start_lb = jsn_op.get('start_lb', 0)
		op.start_ub = jsn_op.get('start_ub', MAX_DUR)

		op.succ = jsn_op['successors']

		for jsn_res in jsn_op.get('resources', []):
			res_idx = self.get_res_idx(jsn_res['resource'])
			res_time = jsn_res.get('release_time', 0)
			
			op.res.append(res_idx)
			train.res_to_op[res_idx].append(op.i)
			self.__res_time[res_idx][op.idx] = res_time

		train.ops.append(op)


	def parse_json_obj(self, jsn_obj):
		if jsn_obj['type'] != 'op_delay':
			return
		
		obj = Obj(
			threshold	=jsn_obj.get('threshold', 0),
			coeff		=jsn_obj.get('coeff', 0),
			increment	=jsn_obj.get('increment', 0)
		)

		self.trains[jsn_obj['train']].ops[jsn_obj['operation']].obj = obj


	def make_prev_ops(self):
		for train in self.trains:
			for op in train.ops:
				for succ in op.succ:
					train.ops[succ].prev.append(op.i)


	def get_res_idx(self, name):
		idx = self.__res_name_idx.get(name, -1)
		
		if idx == -1:
			idx = len(self.__res_name_idx)
			self.__res_name_idx[name] = idx
			self.__res_time.append({})

		return idx


	def calculate_max_dur(self):
		for train in self.trains:
			self.calculate_train_max_dur(train)
		
		self.max_dur = sum(t.max_dur for t in self.trains)

		for train in self.trains:
			for op in train.ops:
				if op.start_ub == MAX_DUR:
					op.start_ub = self.max_dur - op.dur


	def calculate_train_max_dur(self, train: Train):
		n_prev = [op.n_prev for op in train.ops]
		start = [0] * train.n_ops

		q = deque([0])

		while q:
			idx = q.popleft()
			op = train.ops[idx]

			st = op.start_lb
			for prev in op.prev:
				st = max(st, start[prev] + train.ops[prev].dur)

			start[idx] = st

			for succ in op.succ:
				n_prev[succ] -= 1

				if n_prev[succ] == 0:
					q.append(succ)

		train.max_dur = max(start[op.i] + op.dur for op in train.ops if op.n_succ == 0)


	def calculate_avoidable_resources(self):
		for train in self.trains:
			train.avoidable_res = [r for r in train.res if self.is_resource_avoidable(train, r)]


	def is_resource_avoidable(self, train: Train, res: int):
		if res in train.ops[0].res:
			return False

		g = nx.DiGraph()
		g.add_nodes_from(range(train.n_ops + 1))

		for op in train.ops:
			if op.succ:
				g.add_edges_from((op.i, s) for s in op.succ)
			else:
				g.add_edge(op.i, train.n_ops)

		for op in train.ops:
			if res in op.res:
				g.remove_node(op.i)

		return nx.has_path(g, 0, train.n_ops)	


	def op(self, idx: Op_idx) -> Op:
		return self.trains[idx.train].ops[idx.op]
	

	def res_time(self, res_idx: int, op_idx: Op_idx, min_time: int = 0) -> int:
		return min(self.__res_time[res_idx][op_idx], min_time)


	@cached_property
	def n_trains(self):
		return len(self.trains)

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	
	for train in inst.trains:
		print(train.idx, train.avoidable_res)
