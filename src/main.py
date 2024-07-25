from analysis.spectre_utils import Comm20SamplingMetrics
import hydra
import torch
from datasets.dataset import SampledDataModule, SampledDatasetInfo
from datasets.fair_rw import FairRW
from analysis.visualization import NonMolecularVisualization
from model.digress.diffusion_model_discrete import DiscreteDenoisingDiffusion
from model.digress.diffusion_model import LiftedDenoisingDiffusion
from model.digress.diffusion.extra_features import DummyExtraFeatures
from omegaconf import DictConfig
import model.digress.utils as utils
from pytorch_lightning import Trainer
from model.digress.metrics.abstract_metrics import TrainAbstractMetrics


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if torch.cuda.is_available():
        print("Cuda available")
    else:
        print("Cuda not available")

    print(torch.__version__)
    datamodule = SampledDataModule(cfg)
    dataset_infos = SampledDatasetInfo(datamodule, cfg)
    extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
    model_kwargs = {
        'dataset_infos':       dataset_infos,
        'train_metrics':       TrainAbstractMetrics(),
        'sampling_metrics':    Comm20SamplingMetrics(datamodule),
        'visualization_tools': NonMolecularVisualization(),
        'extra_features':      extra_features,
        'domain_features':     domain_features

    }
    
    model = LiftedDenoisingDiffusion(cfg, **model_kwargs)
    callbacks = []

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    torch.set_float32_matmul_precision('high')

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])
    

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()



## Link - Conditional Generation, Fair Sampling, see if uniform and marginal change anything
## Node - Change by loss function by degree, but also change by eigenvector centrality, can I do anything with conditional generation as well?
## Unfairness sources - communities can be mistreated??? nothing to do with individual nodes, this part of the graph as higher loss function???
## Change model - I want node representations and then work with that. also can I also learn the sampler??? would be cool no? then at the end I can say that nodes with similar representations are the same node.
## Learn the sampler and the assembler would be goated.
## Just generating structure, do I need to generate attributes or no, main reason why I want to see code of FAG^2AN
##Also I only generate graphs with the same size of input graph, this is what I think they also do in FAG^2AN
##Graph reproduction, I am just learning that exact graph and not the underlying generative process: see if it is true + edges exist in all subgraphs or they do not exist, maybe mix. edge sampling to solve this problem, also this is intersesting on netgan