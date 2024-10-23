import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer_SiamKPConv import Trainer


# @hydra.main(config_path="conf/configUrb3D.yaml")
@hydra.main(config_path="conf", config_name="configSiamKPConv-cls")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if "deterministic_seed" not in cfg:
        cfg.deterministic_seed = None
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.eval()
    # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
