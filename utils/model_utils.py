from omegaconf import DictConfig


class ModelParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """

    dropout: float
    num_hid_mlp: int
    num_hid_cnn: int
    exp_type: str
    pooling: int
    conv_params: dict
    res: int

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.dropout = kwargs['dropout']
        self.num_hid_mlp = kwargs['num_hid_mlp']
        self.num_hid_cnn = kwargs['num_hid_cnn']
        self.exp_type = kwargs['exp_type']
        self.pooling = kwargs['pooling']
        self.conv_params = kwargs['conv_params']
        self.res = kwargs['res']



def get_model_params(cfg: DictConfig) -> ModelParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return ModelParams(**cfg['model'])
