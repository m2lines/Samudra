# TODO: Remove this file once standalone inference is implemented
import numpy as np
import torch

from utils.data import Normalize


def generate_model_rollout(
    N_eval, test_data, model, hist, N_out, N_extra, initial_input=None, train=False
):
    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[2:], N_out))

    with torch.no_grad():
        outs = model.inference(
            test_data,
            initial_input,
            num_steps=N_eval // (hist + 1),
        )
    for i in range(N_eval // (hist + 1)):
        pred_temp = outs[i]
        pred_temp = torch.nan_to_num(pred_temp)
        pred_temp = torch.clip(pred_temp, min=-1e5, max=1e5)
        C, H, W = pred_temp.shape
        pred_temp = torch.reshape(pred_temp, (hist + 1, C // (hist + 1), H, W))
        model_pred[i * (hist + 1) : (i + 1) * (hist + 1)] = torch.swapaxes(
            torch.swapaxes(pred_temp, 3, 1), 2, 1
        ).cpu()

    if train:
        return model_pred
    else:
        return (
            Normalize.get_instance().unnormalize_numpy_outputs(model_pred),
            outs,
        )
