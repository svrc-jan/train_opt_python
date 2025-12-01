#!.venv/bin/python3

import sys
import json

from collections import deque, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Dict, Set, NamedTuple

import networkx as nx

DEFAULT_DATA = 'data/nor1_critical_2.json'

@dataclass
class Base_res:
	name: str = ''
	time: int = 0


@dataclass
class Base_op:
	dur: int = 0
	start_lb: int = 0
	start_ub: int = -1

	succ: List[int] = field(default_factory=list)
	res: List[Base_res] = field(default_factory=list)

	@property
	def n_succ(self):
		return len(self.succ)

	@property
	def n_res(self):
		return len(self.res)


@dataclass
class Base_obj:
	train: int = -1
	op: int = -1
	threshold: int = 0
	coeff: int = 0
	increment: int = 0


@dataclass
class Base_train:
	ops: List[Base_op] = field(default_factory=list)

	@property
	def n_ops(self):
		return len(self.ops)

class Base_inst:
	trains: List[Base_train]
	objs: List[Base_obj]


	def __init__(self, jsn_file: str):
		self.parse_json_file(jsn_file)


	def parse_json_file(self, jsn_file: str):
		self.trains = []
		self.objs = []

		with open(jsn_file, 'r') as fd:
			jsn = json.load(fd)
		
		for jsn_train in jsn['trains']:
			self.parse_json_train(jsn_train)

		for jsn_obj in jsn['objective']:
			self.parse_json_obj(jsn_obj)


	def parse_json_train(self, jsn_train: dict):
		train = Base_train()
		
		for jsn_op in jsn_train:
			self.parse_json_op(jsn_op, train)

		self.trains.append(train)

		# check if only last op is ending op (n_succ == 0), required for solver
		for op in train.ops[:-1]:
			assert(op.n_succ > 0)

		assert(train.ops[-1].n_succ == 0)


	def parse_json_op(self, jsn_op: dict, train: Base_train):
		op = Base_op(
			dur		=jsn_op['min_duration'],
			start_lb=jsn_op.get('start_lb', 0),
			start_ub=jsn_op.get('start_ub', -1),
			succ = jsn_op['successors']
		)

		for jsn_res in jsn_op.get('resources', []):
			res_name = jsn_res['resource']
			res_time = jsn_res.get('release_time', 0)
			
			op.res.append(Base_res(name=res_name, time=res_time))

		train.ops.append(op)


	def parse_json_obj(self, jsn_obj):
		if jsn_obj['type'] != 'op_delay':
			return
		
		obj = Base_obj(
			train		=jsn_obj['train'],
			op			=jsn_obj['operation'],
			threshold	=jsn_obj.get('threshold', 0),
			coeff		=jsn_obj.get('coeff', 0),
			increment	=jsn_obj.get('increment', 0)
		)

		if obj.coeff == 0 and obj.increment == 0:
			return
		
		assert(obj.coeff == 0 or obj.increment == 0)
		self.objs.append(obj)


	@property
	def n_trains(self):
		return len(self.trains)


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Base_inst(data)
	pass
