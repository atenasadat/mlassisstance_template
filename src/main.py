from importlib import import_module
import sys
from typing import Type
from mlassistant.main import run_main
from mlassistant.entrypoint import BaseEntryPoint


r''' CHANGE `project_name` to your project root package'''
EntryPoint: Type[BaseEntryPoint] = import_module(f'project_name.entrypoints.{sys.argv[1]}').EntryPoint
del sys.argv[1]
ep = EntryPoint()
run_main(ep.conf, ep.model, ep.parser)
