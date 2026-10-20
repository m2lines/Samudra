import dask,torch,time,json,hashlib,gc
from pathlib import Path
from samudra.config import TrainConfig
from samudra.datasets import TrainingWindows,TorchTrainDataset,HostBatch,LoadStats
from torch.utils.data import DataLoader,Subset

class DedupDataset(TorchTrainDataset):
 def __getitem__(self,idx):
  start=time.perf_counter();out=HostBatch(self.id);cache={}
  for step in self.windows.window_plan([idx]).steps:
   uses=(step.input,step.boundary,step.label)
   keys=[(id(u.source),tuple(u.request.time_indices[0]),repr(u.request.channels)) for u in uses]
   pending={}
   for key,use in zip(keys,uses):
    if key not in cache and key not in pending:
     pending[key]=self._get_executor().submit(use.source.read,use.request.time_indices[0],use.request.channels)
   for key,future in pending.items():cache[key]=future.result()
   out.append(*(torch.from_numpy(cache[k].copy()) for k in keys))
  out.load_stats=LoadStats(time.perf_counter()-start)
  return out

root=Path('/scratch/jr7309/om4-current-structure-full-20261020')
dask.config.set(scheduler='synchronous');torch.set_num_threads(1)
cfg=TrainConfig.from_yaml(root/'configs/full-local.yaml')
b=cfg.data.build(data_root=cfg.experiment.resolved_data_root,include_inference=False);l=b.data_layout
w=TrainingWindows(input_source=b.train_sources[0],label_source=None,prognostic_var_names=l.prognostic_var_names,boundary_var_names=l.boundary_var_names,input_steps=2,output_steps=2,steps=4,normalize_before_mask=True,masked_fill_value=0.,stride=1)
plan=w.window_plan([0]);keys=[(id(u.source),tuple(u.request.time_indices[0]),repr(u.request.channels)) for step in plan.steps for u in (step.input,step.boundary,step.label)]
print('read_requests',len(keys),'unique',len(set(keys)),flush=True)
records=[];reference=None
for name,cls in [('baseline',TorchTrainDataset),('dedup',DedupDataset),('baseline_repeat',TorchTrainDataset),('dedup_repeat',DedupDataset)]:
 loader=DataLoader(Subset(cls(w,concurrent_compute_=True),[i*67 for i in range(32)]),batch_size=None,num_workers=8,prefetch_factor=1)
 start=time.perf_counter();digests=[];times=[]
 for batch in loader:
  h=hashlib.sha256()
  for step in batch.steps:
   for tensor in step:h.update(tensor.numpy().tobytes())
  digests.append(h.hexdigest());times.append(time.perf_counter()-start)
 if reference is None:reference=digests
 assert digests==reference,'Tensor mismatch'
 rec={'variant':name,'seconds':time.perf_counter()-start,'steady_seconds_per_sample':(times[-1]-times[15])/16,'identical':True}
 records.append(rec);print(json.dumps(rec),flush=True)
 del loader;gc.collect()
(root/'loader-dedup.json').write_text(json.dumps(records,indent=2))
