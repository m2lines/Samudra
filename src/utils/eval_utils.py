import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import numpy.fft as fft

from .climate_utils import *
from .subgrid_utils import *
from .data_utils import *


def generate_recunet_rollout(N_eval, test_data, model, Nb, Nin):
    model.eval()
    _, _, H, W = test_data.input.shape
    model_pred = np.zeros((N_eval, H, W, Nin))
    # model.verbose = True
    for i in range(N_eval // model.output_time_dim):
        with torch.no_grad():
            if i == 0:
                pred_temp = model(torch.unsqueeze(test_data[i][0], 0), inference=True)
            else:
                pred_temp = model(
                    torch.unsqueeze(test_data[i][0], 0),
                    last_outputs=pred_temp,
                    inference=True,
                )

    final_pred_temp = torch.cat(pred_temp, dim=model.time_dim).squeeze(0)
    print("Final pred temp shape: ", final_pred_temp.shape)
    # Re-add boundaries
    test_target_flat = test_data.input[test_data.output_offset :].reshape(-1, Nin, H, W)
    num_set = final_pred_temp.shape[0]
    final_pred_temp[:, :, :Nb, :] = test_target_flat[:num_set, :Nin, :Nb, :]
    final_pred_temp[:, :, -Nb:, :] = test_target_flat[:num_set, :Nin, -Nb:, :]
    final_pred_temp[:, :, :, :Nb] = test_target_flat[:num_set, :Nin, :, :Nb]
    final_pred_temp[:, :, :, -Nb:] = test_target_flat[:num_set, :Nin, :, -Nb:]

    model_pred = final_pred_temp.reshape(N_eval, H, W, Nin).cpu()
    model_pred = torch.nan_to_num(model_pred)
    model_pred = torch.clip(model_pred, min=-1e5, max=1e5)

    print(model_pred[-1, :5, :5, :])

    return (
        model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"]
    )  # Ideally


def recur_pred(N_eval, test_data, model, hist, N_in, N_extra):

    N_test = test_data.size

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))
    for i in range(N_eval):
        if i == 0:
            with torch.no_grad():
                pred_temp = torch.squeeze(model(torch.unsqueeze(test_data[0][0], 0)))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[1][0][N_in:]), 0)

        elif hist == 0:
            with torch.no_grad():
                pred_temp = torch.squeeze(model(torch.unsqueeze(in_temp, 0)))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if i + 1 < N_test:
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[i + 1][0][N_in:]), 0)
            else:
                ind = np.random.randint(N_test - 1)
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[ind][0][N_in:]), 0)
        else:
            with torch.no_grad():
                pred_temp = torch.squeeze(model((torch.unsqueeze(in_temp, 0))))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if (i > 1) and (hist == 1):
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra)],
                            pred_temp,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra)],
                            pred_temp_old,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
            else:
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
    return model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"]


def recur_pred_joint(N_eval, test_data, model, model_res, hist, N_in, N_extra, N_res):

    N_test = test_data.size

    slices = list(range(N_in))

    for i in range(N_in + N_res, N_in + N_res + N_extra):
        slices.append(i)

    for i in range(N_in + N_res + N_extra, N_in + N_res + N_extra + hist * N_in):
        slices.append(i)

    res_inds = []

    for i in range(N_in, N_in + N_res):
        res_inds.append(i)

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))

    for i in range(N_eval):
        if i == 0:
            with torch.no_grad():
                inpt = test_data[0][0]
                res = model_res(torch.unsqueeze(inpt[slices], 0)).squeeze()
                inpt[res_inds] = torch.squeeze(res)

                pred_temp = torch.squeeze(model(torch.unsqueeze(inpt, 0)))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[1][0][N_in:]), 0)

        elif hist == 0:
            with torch.no_grad():
                pred_temp = torch.squeeze(model(torch.unsqueeze(in_temp, 0)))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if i + 1 < N_test:
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[i + 1][0][N_in:]), 0)
            else:
                ind = np.random.randint(N_test - 1)
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[ind][0][N_in:]), 0)
        else:
            with torch.no_grad():
                res = model_res(torch.unsqueeze(in_temp[slices], 0)).squeeze()
                in_temp[res_inds] = torch.squeeze(res)
                pred_temp = torch.squeeze(model((torch.unsqueeze(in_temp, 0))))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if (i > 1) and (hist == 1):
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp_old,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
            else:
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra + N_res) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra + N_res) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
    return model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"]


def recur_pred_joint_res(
    N_eval, test_data, model, model_res, hist, N_in, N_extra, N_res
):

    N_test = test_data.size

    slices = list(range(N_in))

    for i in range(N_in, N_in + N_extra):
        slices.append(i)

    for i in range(N_in + N_res + N_extra, N_in + N_res + N_extra + hist * N_in):
        slices.append(i)

    res_inds = []

    for i in range(N_in + N_extra, N_in + N_res + N_extra):
        res_inds.append(i)

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))
    res_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))

    for i in range(N_eval):
        if i == 0:
            with torch.no_grad():
                inpt = test_data[0][0]
                res = model_res(torch.unsqueeze(inpt[slices], 0)).squeeze()
                res_pred[i] = torch.swapaxes(torch.swapaxes(res, 2, 0), 1, 0).cpu()
                inpt[res_inds] = torch.squeeze(res)

                pred_temp = torch.squeeze(model(torch.unsqueeze(inpt, 0)))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[1][0][N_in:]), 0)

        elif hist == 0:
            with torch.no_grad():
                pred_temp = torch.squeeze(model(torch.unsqueeze(in_temp, 0)))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if i + 1 < N_test:
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[i + 1][0][N_in:]), 0)
            else:
                ind = np.random.randint(N_test - 1)
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[ind][0][N_in:]), 0)
        else:
            with torch.no_grad():
                res = model_res(torch.unsqueeze(in_temp[slices], 0)).squeeze()
                res_pred[i] = torch.swapaxes(torch.swapaxes(res, 2, 0), 1, 0).cpu()

                in_temp[res_inds] = torch.squeeze(res)
                pred_temp = torch.squeeze(model((torch.unsqueeze(in_temp, 0))))
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if (i > 1) and (hist == 1):
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp_old,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
            else:
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra + N_res) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra + N_res)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra + N_res) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
    return (
        model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"],
        res_pred,
    )


def recur_pred_lateral(N_eval, test_data, model, hist, N_in, N_extra, Nb):

    N_test = test_data.size

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))
    for i in range(N_eval):
        if i == 0:
            with torch.no_grad():
                pred_temp = torch.squeeze(model(torch.unsqueeze(test_data[0][0], 0)))
                if N_in == 1:
                    pred_temp = torch.unsqueeze(pred_temp[0][0], 0)
                pred_temp[:, :Nb, :] = test_data[0][1][:, :Nb, :]
                pred_temp[:, -Nb:, :] = test_data[0][1][:, -Nb:, :]
                pred_temp[:, :, :Nb] = test_data[0][1][:, :, :Nb]
                pred_temp[:, :, -Nb:] = test_data[0][1][:, :, -Nb:]
                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[1][0][N_in:]), 0)

        elif hist == 0:
            with torch.no_grad():
                pred_temp = torch.squeeze(model(torch.unsqueeze(in_temp, 0)))
                if N_in == 1:
                    pred_temp = torch.unsqueeze(pred_temp[0][0], 0)
                pred_temp[:, :Nb, :] = test_data[i][1][:, :Nb, :]
                pred_temp[:, -Nb:, :] = test_data[i][1][:, -Nb:, :]
                pred_temp[:, :, :Nb] = test_data[i][1][:, :, :Nb]
                pred_temp[:, :, -Nb:] = test_data[i][1][:, :, -Nb:]

                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if i + 1 < N_test:
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[i + 1][0][N_in:]), 0)
            else:
                ind = np.random.randint(N_test - 1)
                pred_temp_old = torch.clone(pred_temp)
                in_temp = torch.concat((pred_temp, test_data[ind][0][N_in:]), 0)
        else:
            with torch.no_grad():
                pred_temp = torch.squeeze(model((torch.unsqueeze(in_temp, 0))))
                if N_in == 1:
                    pred_temp = torch.unsqueeze(pred_temp[0][0], 0)
                pred_temp[:, :Nb, :] = test_data[i][1][:, :Nb, :]
                pred_temp[:, -Nb:, :] = test_data[i][1][:, -Nb:, :]
                pred_temp[:, :, :Nb] = test_data[i][1][:, :, :Nb]
                pred_temp[:, :, -Nb:] = test_data[i][1][:, :, -Nb:]

                model_pred[i] = torch.swapaxes(
                    torch.swapaxes(pred_temp, 2, 0), 1, 0
                ).cpu()

            if (i > 1) and (hist == 1):
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra)],
                            pred_temp,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra)],
                            pred_temp_old,
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
            else:
                if i + 1 < N_test:
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[i + 1][0][N_in : (N_in + N_extra)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
                else:
                    ind = np.random.randint(N_test - 1)
                    in_temp = torch.concat(
                        (
                            pred_temp,
                            test_data[ind][0][N_in : (N_in + N_extra)],
                            pred_temp_old,
                            in_temp[(N_in + N_extra) : -N_in],
                        ),
                        0,
                    )
                    pred_temp_old = torch.clone(pred_temp)
    return model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"]


def recur_pred_lateral_norecunet(N_eval, test_data, model, hist, N_in, N_extra, Nb):

    N_test = test_data.size
    assert hist == 0

    img_size = [*test_data[0][0].shape[1:]]
    from .train_utils import get_resize_func

    resize = get_resize_func(img_size)

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))

    # inp = []
    # for step in range(N_eval):
    #     inp.append(resize(test_data[int(step)][0]).cuda())

    # with torch.no_grad():
    #     outs = model(inp)

    with torch.no_grad():
        outs = model.inference(test_data, resize, img_size, num_steps=N_eval)

    for i in range(N_eval):
        pred_temp = outs[i][:, : img_size[0], : img_size[1]]
        pred_temp[:, :Nb, :] = test_data[i][1][:, :Nb, :]
        pred_temp[:, -Nb:, :] = test_data[i][1][:, -Nb:, :]
        pred_temp[:, :, :Nb] = test_data[i][1][:, :, :Nb]
        pred_temp[:, :, -Nb:] = test_data[i][1][:, :, -Nb:]

        pred_temp = torch.nan_to_num(pred_temp)
        pred_temp = torch.clip(pred_temp, min=-1e5, max=1e5)

        model_pred[i] = torch.swapaxes(torch.swapaxes(pred_temp, 2, 0), 1, 0).cpu()

    print(model_pred[-1, :5, :5, 0])
    return model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"]


def generate_unet_rollout(N_eval, test_data, model, hist, N_in, N_extra, Nb):

    N_test = test_data.size
    assert hist == 0

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[1:], N_in))

    with torch.no_grad():
        outs = model.inference(test_data, num_steps=N_eval)

    for i in range(N_eval):
        pred_temp = outs[i]
        pred_temp[:, :Nb, :] = test_data[i][1][:, :Nb, :]
        pred_temp[:, -Nb:, :] = test_data[i][1][:, -Nb:, :]
        pred_temp[:, :, :Nb] = test_data[i][1][:, :, :Nb]
        pred_temp[:, :, -Nb:] = test_data[i][1][:, :, -Nb:]

        pred_temp = torch.nan_to_num(pred_temp)
        pred_temp = torch.clip(pred_temp, min=-1e5, max=1e5)

        model_pred[i] = torch.swapaxes(torch.swapaxes(pred_temp, 2, 0), 1, 0).cpu()

    print(model_pred[-1, :5, :5, 0])
    return model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"]


def compute_corrs(N_eval, test_data, model_pred, wet):
    N_in = model_pred.shape[-1]
    corrs = np.zeros((N_eval, N_in))
    auto_corrs = np.zeros((N_eval, N_in))

    data_out_cpu = test_data[:][1].cpu() * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    data_in_cpu = np.array(test_data[:][0][0][:3].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [1, 2]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [1, 2])

    for i in range(N_eval):
        cor_u = np.corrcoef(
            model_pred[i, wet, 0].flatten(), data_out_cpu[i, 0, wet].flatten()
        )
        cor_v = np.corrcoef(
            model_pred[i, wet, 1].flatten(), data_out_cpu[i, 1, wet].flatten()
        )
        cor_T = np.corrcoef(
            model_pred[i, wet, 2].flatten(), data_out_cpu[i, 2, wet].flatten()
        )

        corrs[i, 0] = cor_u[0, 1]
        corrs[i, 1] = cor_v[0, 1]
        corrs[i, 2] = cor_T[0, 1]

        autocor_u = np.corrcoef(
            data_in_cpu[0, wet].flatten(), data_out_cpu[i, 0, wet].flatten()
        )
        autocor_v = np.corrcoef(
            data_in_cpu[1, wet].flatten(), data_out_cpu[i, 1, wet].flatten()
        )
        autocor_T = np.corrcoef(
            data_in_cpu[2, wet].flatten(), data_out_cpu[i, 2, wet].flatten()
        )

        auto_corrs[i, 0] = autocor_u[0, 1]
        auto_corrs[i, 1] = autocor_v[0, 1]
        auto_corrs[i, 2] = autocor_T[0, 1]

    return corrs, auto_corrs


def compute_corrs_single(N_eval, test_data, model_pred, wet):
    N_in = model_pred.shape[-1]
    corrs = np.zeros((N_eval))
    auto_corrs = np.zeros((N_eval))

    for i in range(N_eval):
        cor_u = np.corrcoef(
            model_pred[i, wet].flatten(), data_out_cpu[i, wet].flatten()
        )

        corrs[i, 0] = cor_u[0, 1]

        autocor_u = np.corrcoef(
            test_data[0, wet].flatten(), test_data[i, wet].flatten()
        )

        auto_corrs[i, 0] = autocor_u[0, 1]

    return corrs, auto_corrs


def compute_corrs_area(N_eval, test_data, model_pred, area, wet):
    model_pred = model_pred.copy()
    N_in = model_pred.shape[-1]
    corrs = np.zeros((N_eval, N_in))
    auto_corrs = np.zeros((N_eval, N_in))
    for i in range(3):
        model_pred[:, :, :, i] = (
            model_pred[:, :, :, i] - test_data.norm_vals["m_out"][i]
        ) / test_data.norm_vals["s_out"][i]
    data_out_cpu = np.array(
        test_data[:][1].cpu()
    )  # *np.expand_dims(test_data.norm_vals['s_out'],[0,2,3])  + np.expand_dims(test_data.norm_vals['m_out'],[0,2,3])
    data_in_cpu_temp = np.array(
        test_data[:][0][0][:3].cpu()
    )  # *np.expand_dims(test_data.norm_vals['s_out'],[1,2])  + np.expand_dims(test_data.norm_vals['m_out'],[1,2])
    data_in_cpu = data_in_cpu_temp.copy()
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):

        cor_u = (
            area_flat
            * model_pred[i, wet, 0].flatten()
            * data_out_cpu[i, 0, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet, 0].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 0, wet].flatten() ** 2).sum()
        )
        cor_v = (
            area_flat
            * model_pred[i, wet, 1].flatten()
            * data_out_cpu[i, 1, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet, 1].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 1, wet].flatten() ** 2).sum()
        )
        cor_T = (
            area_flat
            * model_pred[i, wet, 2].flatten()
            * data_out_cpu[i, 2, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet, 2].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 2, wet].flatten() ** 2).sum()
        )
        corrs[i, 0] = cor_u
        corrs[i, 1] = cor_v
        corrs[i, 2] = cor_T

        autocor_u = (
            area_flat
            * data_in_cpu[0, wet].flatten()
            * data_out_cpu[i, 0, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[0, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 0, wet].flatten() ** 2).sum()
        )
        autocor_v = (
            area_flat
            * data_in_cpu[1, wet].flatten()
            * data_out_cpu[i, 1, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[1, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 1, wet].flatten() ** 2).sum()
        )
        autocor_T = (
            area_flat
            * data_in_cpu[2, wet].flatten()
            * data_out_cpu[i, 2, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[2, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 2, wet].flatten() ** 2).sum()
        )

        auto_corrs[i, 0] = autocor_u
        auto_corrs[i, 1] = autocor_v
        auto_corrs[i, 2] = autocor_T
    return corrs, auto_corrs


def compute_corrs_area_auto(N_eval, test_data, area, wet):
    auto_corrs = np.zeros((N_eval, 3))
    data_out_cpu = np.array(test_data[:][1].cpu())
    data_in_cpu_temp = np.array(test_data[:][0][0][:3].cpu())
    data_in_cpu = data_in_cpu_temp.copy()
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        autocor_u = (
            area_flat
            * data_in_cpu[0, wet].flatten()
            * data_out_cpu[i, 0, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[0, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 0, wet].flatten() ** 2).sum()
        )
        autocor_v = (
            area_flat
            * data_in_cpu[1, wet].flatten()
            * data_out_cpu[i, 1, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[1, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 1, wet].flatten() ** 2).sum()
        )
        autocor_T = (
            area_flat
            * data_in_cpu[2, wet].flatten()
            * data_out_cpu[i, 2, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[2, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 2, wet].flatten() ** 2).sum()
        )

        auto_corrs[i, 0] = autocor_u
        auto_corrs[i, 1] = autocor_v
        auto_corrs[i, 2] = autocor_T
    return auto_corrs


def compute_ACC(N_eval, test_data, model_pred, clim, time, area, wet):
    model_pred = model_pred.copy()
    clim = clim.copy()
    N_in = model_pred.shape[-1]
    corrs = np.zeros((N_eval, N_in))
    auto_corrs = np.zeros((N_eval, N_in))
    for i in range(3):
        model_pred[:, :, :, i] = (
            model_pred[:, :, :, i] - test_data.norm_vals["m_out"][i]
        ) / test_data.norm_vals["s_out"][i]
        clim[:, :, :, i] = (
            clim[:, :, :, i] - test_data.norm_vals["m_out"][i]
        ) / test_data.norm_vals["s_out"][i]
    data_out_cpu = np.array(
        test_data[:][1].cpu()
    )  # *np.expand_dims(test_data.norm_vals['s_out'],[0,2,3])  + np.expand_dims(test_data.norm_vals['m_out'],[0,2,3])
    data_in_cpu_temp = np.array(
        test_data[:][0][0][:3].cpu()
    )  # *np.expand_dims(test_data.norm_vals['s_out'],[1,2])  + np.expand_dims(test_data.norm_vals['m_out'],[1,2])
    data_in_cpu = data_in_cpu_temp.copy()
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        day = int(time[i].dayofyr - 1)
        for j in range(N_in):
            model_pred[i, :, :, j] -= clim[day, :, :, j].squeeze()
            data_out_cpu[i, j] -= clim[day, :, :, j].squeeze()
            data_in_cpu[j] = data_in_cpu_temp[j] - clim[day, :, :, j].squeeze()
        cor_u = (
            area_flat
            * model_pred[i, wet, 0].flatten()
            * data_out_cpu[i, 0, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet, 0].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 0, wet].flatten() ** 2).sum()
        )
        cor_v = (
            area_flat
            * model_pred[i, wet, 1].flatten()
            * data_out_cpu[i, 1, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet, 1].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 1, wet].flatten() ** 2).sum()
        )
        cor_T = (
            area_flat
            * model_pred[i, wet, 2].flatten()
            * data_out_cpu[i, 2, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet, 2].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 2, wet].flatten() ** 2).sum()
        )

        corrs[i, 0] = cor_u
        corrs[i, 1] = cor_v
        corrs[i, 2] = cor_T

        autocor_u = (
            area_flat
            * data_in_cpu[0, wet].flatten()
            * data_out_cpu[i, 0, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[0, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 0, wet].flatten() ** 2).sum()
        )
        autocor_v = (
            area_flat
            * data_in_cpu[1, wet].flatten()
            * data_out_cpu[i, 1, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[1, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 1, wet].flatten() ** 2).sum()
        )
        autocor_T = (
            area_flat
            * data_in_cpu[2, wet].flatten()
            * data_out_cpu[i, 2, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[2, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 2, wet].flatten() ** 2).sum()
        )

        auto_corrs[i, 0] = autocor_u
        auto_corrs[i, 1] = autocor_v
        auto_corrs[i, 2] = autocor_T

    return corrs, auto_corrs


def compute_ACC_auto(N_eval, test_data, clim, time, area, wet):
    clim = clim.copy()
    N_in = 3
    auto_corrs = np.zeros((N_eval, N_in))
    for i in range(3):
        clim[:, :, :, i] = (
            clim[:, :, :, i] - test_data.norm_vals["m_out"][i]
        ) / test_data.norm_vals["s_out"][i]
    data_out_cpu = np.array(test_data[:][1].cpu())
    data_in_cpu_temp = np.array(test_data[:][0][0][:3].cpu())
    data_in_cpu = data_in_cpu_temp.copy()
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        day = int(time[i].dayofyr - 1)
        for j in range(N_in):
            data_out_cpu[i, j] -= clim[day, :, :, j].squeeze()
            data_in_cpu[j] = data_in_cpu_temp[j] - clim[day, :, :, j].squeeze()

        autocor_u = (
            area_flat
            * data_in_cpu[0, wet].flatten()
            * data_out_cpu[i, 0, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[0, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 0, wet].flatten() ** 2).sum()
        )
        autocor_v = (
            area_flat
            * data_in_cpu[1, wet].flatten()
            * data_out_cpu[i, 1, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[1, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 1, wet].flatten() ** 2).sum()
        )
        autocor_T = (
            area_flat
            * data_in_cpu[2, wet].flatten()
            * data_out_cpu[i, 2, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * data_in_cpu[2, wet].flatten() ** 2).sum()
            * (area_flat * data_out_cpu[i, 2, wet].flatten() ** 2).sum()
        )

        auto_corrs[i, 0] = autocor_u
        auto_corrs[i, 1] = autocor_v
        auto_corrs[i, 2] = autocor_T

    return auto_corrs


def compute_rmse(N_eval, test_data, model_pred, area, wet):
    N_in = model_pred.shape[-1]

    rmse = np.zeros((N_eval, N_in))
    auto_rmse = np.zeros((N_eval, N_in))

    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    data_in_cpu = np.array(test_data[:][0][0][:N_in].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [1, 2]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [1, 2])
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        rmse_u = np.sqrt(
            (
                area_flat
                * (model_pred[i, wet, 0].flatten() - data_out_cpu[i, 0, wet].flatten())
                ** 2
            ).sum()
            / area_flat.sum()
        )
        rmse_v = np.sqrt(
            (
                area_flat
                * (model_pred[i, wet, 1].flatten() - data_out_cpu[i, 1, wet].flatten())
                ** 2
            ).sum()
            / area_flat.sum()
        )
        rmse_T = np.sqrt(
            (
                area_flat
                * (model_pred[i, wet, 2].flatten() - data_out_cpu[i, 2, wet].flatten())
                ** 2
            ).sum()
            / area_flat.sum()
        )

        rmse[i, 0] = rmse_u
        rmse[i, 1] = rmse_v
        rmse[i, 2] = rmse_T

        autormse_u = np.sqrt(
            (
                area_flat
                * (data_in_cpu[0, wet].flatten() - data_out_cpu[i, 0, wet].flatten())
                ** 2
            ).sum()
            / area_flat.sum()
        )
        autormse_v = np.sqrt(
            (
                area_flat
                * (data_in_cpu[1, wet].flatten() - data_out_cpu[i, 1, wet].flatten())
                ** 2
            ).sum()
            / area_flat.sum()
        )
        autormse_T = np.sqrt(
            (
                area_flat
                * (data_in_cpu[2, wet].flatten() - data_out_cpu[i, 2, wet].flatten())
                ** 2
            ).sum()
            / area_flat.sum()
        )

        auto_rmse[i, 0] = autormse_u
        auto_rmse[i, 1] = autormse_v
        auto_rmse[i, 2] = autormse_T

    return rmse, auto_rmse


def compute_mean(N_eval, test_data, model_pred, area, wet):
    N_in = model_pred.shape[-1]

    mean = np.zeros((N_eval, N_in))
    auto_mean = np.zeros((N_eval, N_in))

    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])

    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        mean_u = (area_flat * model_pred[i, wet, 0].flatten()).sum() / area_flat.sum()
        mean_v = (area_flat * model_pred[i, wet, 1].flatten()).sum() / area_flat.sum()
        mean_T = (area_flat * model_pred[i, wet, 2].flatten()).sum() / area_flat.sum()

        mean[i, 0] = mean_u
        mean[i, 1] = mean_v
        mean[i, 2] = mean_T

        automean_u = (
            area_flat * data_out_cpu[i, 0, wet].flatten()
        ).sum() / area_flat.sum()
        automean_v = (
            area_flat * data_out_cpu[i, 1, wet].flatten()
        ).sum() / area_flat.sum()
        automean_T = (
            area_flat * data_out_cpu[i, 2, wet].flatten()
        ).sum() / area_flat.sum()

        auto_mean[i, 0] = automean_u
        auto_mean[i, 1] = automean_v
        auto_mean[i, 2] = automean_T

    return mean, auto_mean


def compute_var(N_eval, test_data, model_pred, area, wet):
    N_in = model_pred.shape[-1]

    mean = np.zeros((N_eval, N_in))
    auto_mean = np.zeros((N_eval, N_in))

    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])

    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        mean_u = (area_flat * model_pred[i, wet, 0].flatten()).sum() / area_flat.sum()
        mean_v = (area_flat * model_pred[i, wet, 1].flatten()).sum() / area_flat.sum()
        mean_T = (area_flat * model_pred[i, wet, 2].flatten()).sum() / area_flat.sum()

        var_u = (
            area_flat * ((model_pred[i, wet, 0].flatten() - mean_u) ** 2)
        ).sum() / area_flat.sum()
        var_v = (
            area_flat * ((model_pred[i, wet, 1].flatten() - mean_v) ** 2)
        ).sum() / area_flat.sum()
        var_T = (
            area_flat * ((model_pred[i, wet, 2].flatten() - mean_T) ** 2)
        ).sum() / area_flat.sum()

        mean[i, 0] = var_u
        mean[i, 1] = var_v
        mean[i, 2] = var_T

        automean_u = (
            area_flat * data_out_cpu[i, 0, wet].flatten()
        ).sum() / area_flat.sum()
        automean_v = (
            area_flat * data_out_cpu[i, 1, wet].flatten()
        ).sum() / area_flat.sum()
        automean_T = (
            area_flat * data_out_cpu[i, 2, wet].flatten()
        ).sum() / area_flat.sum()

        autovar_u = (
            area_flat * ((data_out_cpu[i, 0, wet].flatten() - automean_u) ** 2)
        ).sum() / area_flat.sum()
        autovar_v = (
            area_flat * ((data_out_cpu[i, 1, wet].flatten() - automean_v) ** 2)
        ).sum() / area_flat.sum()
        autovar_T = (
            area_flat * ((data_out_cpu[i, 2, wet].flatten() - automean_T) ** 2)
        ).sum() / area_flat.sum()

        auto_mean[i, 0] = autovar_u
        auto_mean[i, 1] = autovar_v
        auto_mean[i, 2] = autovar_T

    return mean, auto_mean


def compute_auto_var(N_eval, test_data, area, wet):
    N_in = 3

    auto_mean = np.zeros((N_eval, N_in))

    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])

    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):

        automean_u = (
            area_flat * data_out_cpu[i, 0, wet].flatten()
        ).sum() / area_flat.sum()
        automean_v = (
            area_flat * data_out_cpu[i, 1, wet].flatten()
        ).sum() / area_flat.sum()
        automean_T = (
            area_flat * data_out_cpu[i, 2, wet].flatten()
        ).sum() / area_flat.sum()

        autovar_u = (
            area_flat * ((data_out_cpu[i, 0, wet].flatten() - automean_u) ** 2)
        ).sum() / area_flat.sum()
        autovar_v = (
            area_flat * ((data_out_cpu[i, 1, wet].flatten() - automean_v) ** 2)
        ).sum() / area_flat.sum()
        autovar_T = (
            area_flat * ((data_out_cpu[i, 2, wet].flatten() - automean_T) ** 2)
        ).sum() / area_flat.sum()

        auto_mean[i, 0] = autovar_u
        auto_mean[i, 1] = autovar_v
        auto_mean[i, 2] = autovar_T

    return auto_mean


def compute_time_spec(N_eval, test_data, model_pred, lag):
    N_in = test_data.shape[-1]

    freqs = fft.rfftfreq(N_eval, lag)

    ffts = np.zeros((freqs.size, N_in))
    true_ffts = np.zeros((freqs.size, N_in))
    for i in range(N_in):
        true_ffts[:, i] = np.abs(fft.rfft(test_data[:N_eval, i]))
        ffts[:, i] = np.abs(fft.rfft(model_pred[:N_eval, i]))

    return freqs, ffts, true_ffts


def compute_heat_flux(N_eval, test_data, model_pred, dx, dy):
    N_in = model_pred.shape[-1]
    model_pred = model_pred[:N_eval].copy()

    Cw = 4218  # J/(KG K)
    rho = 1020  # Kg/m^3

    data_out_cpu = np.array(test_data[:N_eval][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    data_out_cpu[:, 2] = data_out_cpu[:, 2] + 273.15
    model_pred[:, :, :, 2] = model_pred[:, :, :, 2] + 273.15

    data_out_cpu = np.swapaxes(np.swapaxes(data_out_cpu, 3, 2), 3, 1)

    flux_v = model_pred[:N_eval, :, :, 1] * model_pred[:N_eval, :, :, 2] * Cw * rho * dx
    flux_u = model_pred[:N_eval, :, :, 0] * model_pred[:N_eval, :, :, 2] * Cw * rho * dy

    flux_true_v = (
        data_out_cpu[:N_eval, :, :, 1] * data_out_cpu[:N_eval, :, :, 2] * Cw * rho * dx
    )
    flux_true_u = (
        data_out_cpu[:N_eval, :, :, 0] * data_out_cpu[:N_eval, :, :, 2] * Cw * rho * dy
    )

    return flux_u, flux_v, flux_true_u, flux_true_v


def compute_KE(N_eval, test_data, model_pred, area, wet):
    N_in = model_pred.shape[-1]

    KE = np.zeros((N_eval,))
    auto_KE = np.zeros((N_eval,))

    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])

    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        KE_u = (
            area_flat * (model_pred[i, wet, 0] ** 2).flatten()
        ).sum() / area_flat.sum()
        KE_v = (
            area_flat * (model_pred[i, wet, 1] ** 2).flatten()
        ).sum() / area_flat.sum()

        KE[i] = 0.5 * (KE_u + KE_v)

        autoKE_u = (
            area_flat * (data_out_cpu[i, 0, wet] ** 2).flatten()
        ).sum() / area_flat.sum()
        autoKE_v = (
            area_flat * (data_out_cpu[i, 1, wet] ** 2).flatten()
        ).sum() / area_flat.sum()

        auto_KE[i] = 0.5 * (autoKE_u + autoKE_v)

    return KE, auto_KE


def compute_activity(N_eval, test_data, model_pred, clim, time, area, wet):
    N_in = model_pred.shape[-1]

    activity = np.zeros((N_eval, N_in))
    auto_activity = np.zeros((N_eval, N_in))
    clim = clim.copy()
    model_pred = model_pred.copy()
    N_in = 3
    auto_corrs = np.zeros((N_eval, N_in))
    for i in range(3):
        clim[:, :, :, i] = (
            clim[:, :, :, i] - test_data.norm_vals["m_out"][i]
        ) / test_data.norm_vals["s_out"][i]
        model_pred[i] = (
            model_pred[i] - test_data.norm_vals["m_out"][i]
        ) / test_data.norm_vals["s_out"][i]
    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    data_in_cpu = np.array(test_data[:][0][0][:N_in].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [1, 2]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [1, 2])
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        day = int(time[i].dayofyr - 1)

        mean_error_u = (
            area_flat * (model_pred[i, wet, 0].flatten() - clim[day, wet, 0].flatten())
        ).sum() / area_flat.sum()
        mean_error_v = (
            area_flat * (model_pred[i, wet, 1].flatten() - clim[day, wet, 1].flatten())
        ).sum() / area_flat.sum()
        mean_error_T = (
            area_flat * (model_pred[i, wet, 2].flatten() - clim[day, wet, 2].flatten())
        ).sum() / area_flat.sum()

        act_u = np.sqrt(
            (
                area_flat
                * (
                    (model_pred[i, wet, 0].flatten() - clim[day, wet, 0].flatten())
                    - mean_error_u
                )
                ** 2
            ).sum()
            / area_flat.sum()
        )
        act_v = np.sqrt(
            (
                area_flat
                * (
                    (model_pred[i, wet, 1].flatten() - clim[day, wet, 1].flatten())
                    - mean_error_v
                )
                ** 2
            ).sum()
            / area_flat.sum()
        )
        act_T = np.sqrt(
            (
                area_flat
                * (
                    (model_pred[i, wet, 2].flatten() - clim[day, wet, 2].flatten())
                    - mean_error_T
                )
                ** 2
            ).sum()
            / area_flat.sum()
        )

        activity[i, 0] = act_u
        activity[i, 1] = act_v
        activity[i, 2] = act_T

        auto_me_u = (
            area_flat * (data_out_cpu[i, 0, wet].flatten() - clim[day, wet, 0])
        ).sum() / area_flat.sum()
        auto_me_v = (
            area_flat * (data_out_cpu[i, 1, wet].flatten() - clim[day, wet, 1])
        ).sum() / area_flat.sum()
        auto_me_T = (
            area_flat * (data_out_cpu[i, 2, wet].flatten() - clim[day, wet, 2])
        ).sum() / area_flat.sum()

        auto_act_u = np.sqrt(
            (
                area_flat
                * ((data_out_cpu[i, 0, wet].flatten() - clim[day, wet, 0]) - auto_me_u)
                ** 2
            ).sum()
            / area_flat.sum()
        )
        auto_act_v = np.sqrt(
            (
                area_flat
                * ((data_out_cpu[i, 1, wet].flatten() - clim[day, wet, 1]) - auto_me_v)
                ** 2
            ).sum()
            / area_flat.sum()
        )
        auto_act_T = np.sqrt(
            (
                area_flat
                * ((data_out_cpu[i, 2, wet].flatten() - clim[day, wet, 2]) - auto_me_T)
                ** 2
            ).sum()
            / area_flat.sum()
        )

        auto_activity[i, 0] = auto_act_u
        auto_activity[i, 1] = auto_act_v
        auto_activity[i, 2] = auto_act_T

    return activity, auto_activity


def gen_enstrophy(N_eval, test_data, model_pred, dx, dy, Nb, wet_lap):
    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    pred_vort = compute_vorticity(
        model_pred[:N_eval, :, :, 0], model_pred[:N_eval, :, :, 1], dx, dy, Nb, wet_lap
    )
    pred_enst = pred_vort**2
    true_vort = compute_vorticity(
        data_out_cpu[:N_eval, 0], data_out_cpu[:N_eval, 1], dx, dy, Nb, wet_lap
    )
    true_enst = true_vort**2
    return pred_enst, true_enst


def gen_vorticity(N_eval, test_data, model_pred, dx, dy, Nb, wet_lap):
    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    pred_vort = compute_vorticity(
        model_pred[:N_eval, :, :, 0], model_pred[:N_eval, :, :, 1], dx, dy, Nb, wet_lap
    )
    true_vort = compute_vorticity(
        data_out_cpu[:N_eval, 0], data_out_cpu[:N_eval, 1], dx, dy, Nb, wet_lap
    )
    return pred_vort, true_vort


def gen_KE(N_eval, test_data, model_pred):
    rho = 1.2e3
    data_out_cpu = np.array(test_data[:][1].cpu()) * np.expand_dims(
        test_data.norm_vals["s_out"], [0, 2, 3]
    ) + np.expand_dims(test_data.norm_vals["m_out"], [0, 2, 3])
    pred_KE = (
        (model_pred[:N_eval, :, :, 0] ** 2 + model_pred[:N_eval, :, :, 1] ** 2)
        * 0.5
        * rho
    )
    true_KE = (
        (data_out_cpu[:N_eval, 0] ** 2 + data_out_cpu[:N_eval, 1] ** 2) * 0.5 * rho
    )
    return pred_KE, true_KE


def compute_corrs_single(N_eval, test_data, model_pred, area, wet, std, mean):
    N_in = model_pred.shape[-1]
    corrs = np.zeros((N_eval))
    test_data = (test_data - mean) / std
    model_pred = (model_pred - mean) / std
    auto_corrs = np.zeros((N_eval))
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):

        cor_u = (
            area_flat * model_pred[i, wet].flatten() * test_data[i, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet].flatten() ** 2).sum()
            * (area_flat * test_data[i, wet].flatten() ** 2).sum()
        )

        corrs[i] = cor_u

        autocor_u = (
            area_flat * test_data[0, wet].flatten() * test_data[i, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * test_data[0, wet].flatten() ** 2).sum()
            * (area_flat * test_data[i, wet].flatten() ** 2).sum()
        )

        auto_corrs[i] = autocor_u

    return corrs, auto_corrs


def compute_RMSE_single(N_eval, test_data, model_pred, area, wet):
    N_in = model_pred.shape[-1]

    rmse = np.zeros((N_eval))
    auto_rmse = np.zeros((N_eval))

    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        rmse_u = np.sqrt(
            (
                area_flat
                * (model_pred[i, wet].flatten() - test_data[i, wet].flatten()) ** 2
            ).sum()
            / area_flat.sum()
        )

        rmse[i] = rmse_u

        autormse_u = np.sqrt(
            (
                area_flat
                * (test_data[0, wet].flatten() - test_data[i, wet].flatten()) ** 2
            ).sum()
            / area_flat.sum()
        )

        auto_rmse[i] = autormse_u

    return rmse, auto_rmse


def compute_ACC_single(N_eval, test_data, model_pred, clim, time, area, wet):
    model_pred = model_pred.copy()
    test_data = test_data.copy()
    clim = clim.copy()
    N_in = model_pred.shape[-1]
    corrs = np.zeros((N_eval))
    auto_corrs = np.zeros((N_eval))
    area_flat = np.array(area[wet].flatten())

    for i in range(N_eval):
        day = int(time[i].dayofyr - 1)
        model_pred[i, :, :] -= clim[day, :, :].squeeze()
        test_data[i] -= clim[day, :, :].squeeze()
        cor_u = (
            area_flat * model_pred[i, wet].flatten() * test_data[i, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * model_pred[i, wet].flatten() ** 2).sum()
            * (area_flat * test_data[i, wet].flatten() ** 2).sum()
        )

        corrs[i] = cor_u

        autocor_u = (
            area_flat * test_data[0, wet].flatten() * test_data[i, wet].flatten()
        ).sum() / np.sqrt(
            (area_flat * test_data[0, wet].flatten() ** 2).sum()
            * (area_flat * test_data[i, wet].flatten() ** 2).sum()
        )

        auto_corrs[i] = autocor_u

    return corrs, auto_corrs


def gen_KE_spectrum(N_eval, test_data, model_pred, grids, wet):
    std_out = test_data.norm_vals["s_out"]
    mean_out = test_data.norm_vals["m_out"]
    u_test = test_data[:][1][:, 0] * std_out[0] + mean_out[0]
    v_test = test_data[:][1][:, 1] * std_out[1] + mean_out[1]

    region_fft = get_domain_fft(wet)
    dx_fft = grids["dxu"][region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]]
    dy_fft = grids["dyu"][region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]]

    KE_spec = KE_spectrum_long(
        dx_fft,
        dy_fft,
        model_pred[
            :N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1], 0
        ],
        model_pred[
            :N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1], 1
        ],
    )
    KE_spec_true = KE_spectrum_long(
        dx_fft,
        dy_fft,
        u_test[:N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]],
        v_test[:N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]],
    )
    return KE_spec, KE_spec_true


def gen_enstrophy_spectrum(N_eval, test_data, model_pred, grids, wet, wet_lap, Nb=4):
    dx = grids["dxu"].to_numpy()
    dy = grids["dyu"].to_numpy()
    pred_vort, true_vort = gen_vorticity(
        1000, test_data, model_pred, dx, dy, 4, wet_lap
    )
    region_fft = get_domain_fft(wet[Nb:-Nb, Nb:-Nb])
    dx_fft = grids["dxu"][Nb:-Nb, Nb:-Nb][
        region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]
    ]
    dy_fft = grids["dyu"][Nb:-Nb, Nb:-Nb][
        region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]
    ]

    enst_spec = KE_spectrum_long(
        dx_fft,
        dy_fft,
        pred_vort[
            :N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]
        ],
        pred_vort[
            :N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]
        ],
    )
    enst_spec_true = KE_spectrum_long(
        dx_fft,
        dy_fft,
        true_vort[
            :N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]
        ],
        true_vort[
            :N_eval, region_fft[2] : region_fft[3], region_fft[0] : region_fft[1]
        ],
    )
    return enst_spec, enst_spec_true
