from .real_mask_model import CNN_RNN_FC_REAL_MASK_MODEL
from . import modules
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class, var = {
      "CNN_RNN_FC_REAL_MASK_MODEL": (CNN_RNN_FC_REAL_MASK_MODEL, modules.RealVariables),
  }[PARAM.model_name]

  return model_class, var
