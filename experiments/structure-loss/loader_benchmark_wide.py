import dask,torch,time,json,hashlib,gc
from pathlib import Path
from samudra.config import TrainConfig
from samudra.datasets import TrainingWindows,TorchTrainDataset
from torch.utils.data import DataLoader,Subset
root=Path('/scratch/jr7309/om4-current-structure-full-20261020')
dask.config.set(scheduler='synchronous');torch.set_num_threads(1)
cfg=TrainConfig.from_yaml(root/'configs/full-local.yaml')
b=cfg.data.build(data_root=cfg.experiment.resolved_data_root,include_inference=False)
l=b.data_layout
w=TrainingWindows(input_source=b.train_sources[0],label_source=None,prognostic_var_names=l.prognostic_var_names,boundary_var_names=l.boundary_var_names,input_steps=2,output_steps=2,steps=4,normalize_before_mask=True,masked_fill_value=0.,stride=1)
records=[];reference=None
indices=[i*67 for i in range(32)]
for workers,concurrent in [(8,True),(16,True)]:
 ds=Subset(TorchTrainDataset(w,concurrent_compute_=concurrent),indices)
 loader=DataLoader(ds,batch_size=None,num_workers=workers,prefetch_factor=1,persistent_workers=False)
 start=time.perf_counter();digests=[];times=[]
 for batch in loader:
  h=hashlib.sha256()
  for step in batch.steps:
   for tensor in step:h.update(tensor.numpy().tobytes())
  digests.append(h.hexdigest());times.append(time.perf_counter()-start)
 if reference is None:reference=digests
 assert digests==reference,'Tensor mismatch'
 record={'workers':workers,'concurrent':concurrent,'seconds':time.perf_counter()-start,'steady_seconds_per_sample':(times[-1]-times[15])/16,'identical':True}
 records.append(record);print(json.dumps(record),flush=True)
 del loader,ds;gc.collect()
(root/'loader-benchmark-wide.json').write_text(json.dumps(records,indent=2))
