from fanjiang.utils.registry import Registry

MODELS = Registry("models")
CRITERIONS = Registry("criterions")
METRICS = Registry("metrics")
DATASETS = Registry("datasets")

def build(cfg, func):
    values = list(cfg.values())
    if len(values) == 0:
        return {}

    module = {}
    if isinstance(values[0], dict):
        for name, cfg_ in cfg.items():
            module[name] = func(cfg_)
        return module
    else:
        return func(cfg)

def build_model(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    model = MODELS.get(name)(**cfg_)
    return model

def build_models(cfg):
    return build(cfg, build_model)

def build_criterion(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    criterion = CRITERIONS.get(name)(**cfg_)
    return criterion

def build_criterions(cfg):
    return build(cfg, build_criterion)

def build_metric(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    metrics = METRICS.get(name)(**cfg_)
    return metrics

def build_metrics(cfg):
    metrics = build(cfg, build_metric)
    return list(metrics.values())

def build_dataset(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    dataset = DATASETS.get(name)(**cfg_)
    return dataset

def build_datasets(cfg):
    datasets = build(cfg, build_dataset)
    datasets = list(datasets.values())
    if len(datasets) == 1:
        return datasets[0]
    return datasets

