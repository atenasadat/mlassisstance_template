from importlib import import_module
import sys
from typing import Type
from mlassistant.main import run_main
from mlassistant.entrypoint import BaseEntryPoint


r''' CHANGE `xray_segmentation` to your project root package'''
EntryPoint: Type[BaseEntryPoint] = import_module(f'xray_segmentation.entrypoints.{sys.argv[1]}').EntryPoint
del sys.argv[1]
ep = EntryPoint()
run_main(ep.conf, ep.model, ep.parser)
