"""CPU consumer-branch probe, NOT a GPU/model parity test.
Executes installed DFlash/DSpark store method with synthetic aux and a stub projection.
A poison last_hidden_states fails on any access, showing the aux branch ignores it.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py')
tree=ast.parse(p.read_text())
fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_store_hidden_states')
for a in fn.args.args: a.annotation=None
fn.returns=None
module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
ns={'torch':torch};exec(compile(module,str(p),'exec'),ns)
class Poison:
 def __getitem__(self,k):raise AssertionError('MTP residual was read')
for staged in (0,2,4):
 aux=[torch.full((4,8),float(i)) for i in range(3)]
 output=torch.full((4,8),float('nan'))
 self=NS(_aux_staging=torch.zeros((4,24)),hidden_states=output,model=NS(combine_hidden_states=lambda x:x.reshape(-1,3,8).sum(1)))
 ns['_store_hidden_states'](self,4,Poison(),aux,staged)
 if staged: assert torch.equal(self._aux_staging[:staged],torch.cat(aux,dim=-1)[:staged])
 assert torch.equal(output[staged:],torch.full((4-staged,8),3.0))
 print('PASS auxiliary branch ignores MTP residual; graph staging rows',staged)
