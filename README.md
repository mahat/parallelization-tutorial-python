# parallelization-tutorial-python
## Introduction
Some scripts for introducing python dist package, and small training strategies i.e. distributed data, batch sync, async training ,optimizer sharding only using torch.dist or nn.Module. Codes are written for presentation/tutorial therefore they are not production ready. There are code duplications for better fitting code to single screen. 
## Files
- *test.py -> small scripts for showcase torch.dist
- dist_zero.py -> sync distributed data training
- dist_zero_async.py -> async distributed data training
- dist_zero_async_bucket.py -> async distributed data training with bucketed communication 
- dist_one.py ->  async distributed data training with bucketed communication and shared optimizer state
## TODOs
- adding profiler
- broadcast