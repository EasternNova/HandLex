from backend.modelA.inference import ModelAInference

_model_a = None


def load_model_a():
    global _model_a

    if _model_a is None:
        _model_a = ModelAInference()

    return _model_a


def get_model_a():
    return _model_a


def unload_model_a():
    global _model_a

    if _model_a is not None:
        _model_a.close()
        _model_a = None