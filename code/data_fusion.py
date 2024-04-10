import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def perform_concat(X_s, X_f, X_t):
    import numpy as np

    # 两两拼接
    X_sf = np.concatenate([X_s, X_f], axis=1)  # S和F拼接
    X_st = np.concatenate([X_s, X_t], axis=1)  # S和T拼接
    X_ft = np.concatenate([X_f, X_t], axis=1)  # F和T拼接

    # 三个一起拼接
    X_stf = np.concatenate([X_s, X_f, X_t], axis=1)  # S，F和T一起拼接

    return X_sf, X_st, X_ft, X_stf


def perform_cpb(X_s, X_f, X_t):
    from compact_bilinear_pooling import CompactBilinearPooling

    X_s_tensor = Variable(torch.Tensor(X_s), requires_grad=True)
    X_f_tensor = Variable(torch.Tensor(X_f), requires_grad=True)
    X_t_tensor = Variable(torch.Tensor(X_t), requires_grad=True)

    # 根据输入和输出维度定义 CPB
    mcb_sf = CompactBilinearPooling(X_s.shape[1], X_f.shape[1], X_s.shape[1] + X_f.shape[1])
    X_sf_tensor = mcb_sf(X_s_tensor, X_f_tensor)

    mcb_st = CompactBilinearPooling(X_s.shape[1], X_t.shape[1], X_s.shape[1] + X_t.shape[1])
    X_st_tensor = mcb_st(X_s_tensor, X_t_tensor)

    mcb_ft = CompactBilinearPooling(X_f.shape[1], X_t.shape[1], X_f.shape[1] + X_t.shape[1])
    X_ft_tensor = mcb_ft(X_f_tensor, X_t_tensor)

    mcb_stf = CompactBilinearPooling(X_s.shape[1], X_f.shape[1] + X_t.shape[1],
                                     X_s.shape[1] + X_f.shape[1] + X_t.shape[1])
    X_stf_tensor = mcb_stf(X_s_tensor, X_ft_tensor)

    return X_sf_tensor.detach().numpy(), X_st_tensor.detach().numpy(), X_ft_tensor.detach().numpy(), X_stf_tensor.detach().numpy()


def perform_tensor_fusion(X_s, X_f, X_t):
    from tensor_fusion import TensorFusion

    X_s_tensor = Variable(torch.Tensor(X_s), requires_grad=True)
    X_f_tensor = Variable(torch.Tensor(X_f), requires_grad=True)
    X_t_tensor = Variable(torch.Tensor(X_t), requires_grad=True)

    # 根据输入和输出维度定义 TensorFusion
    tf_sf = TensorFusion(X_s.shape[1], X_f.shape[1], X_s.shape[1] + X_f.shape[1])
    X_sf_tensor = tf_sf(X_s_tensor, X_f_tensor)

    tf_st = TensorFusion(X_s.shape[1], X_t.shape[1], X_s.shape[1] + X_t.shape[1])
    X_st_tensor = tf_st(X_s_tensor, X_t_tensor)

    tf_ft = TensorFusion(X_f.shape[1], X_t.shape[1], X_f.shape[1] + X_t.shape[1])
    X_ft_tensor = tf_ft(X_f_tensor, X_t_tensor)

    tf_stf = TensorFusion(X_s.shape[1] + X_f.shape[1], X_t.shape[1], X_s.shape[1] + X_f.shape[1] + X_t.shape[1])
    X_stf_tensor = tf_stf(X_sf_tensor, X_t_tensor)

    return X_sf_tensor.detach().numpy(), X_st_tensor.detach().numpy(), X_ft_tensor.detach().numpy(), X_stf_tensor.detach().numpy()


def process_and_save_data_general(satellite_path, textual_path, financial_path, save_path, fusion_method):

    # Load datasets
    satellite_data = pd.read_excel(satellite_path).iloc[:, 5:]
    textual_data = pd.read_excel(textual_path).iloc[:, 5:]
    financial_data = pd.read_excel(financial_path).iloc[:, 5:]
    default = financial_data.loc[:, 'default'].values
    stf_data = np.concatenate([satellite_data, financial_data.iloc[:, 5:], textual_data], axis=1)

    # Split and oversample data
    X_train, X_test, y_train, y_test = train_test_split(stf_data, default, test_size=0.3, random_state=42,
                                                        stratify=default)
    oversample = RandomOverSampler(sampling_strategy=1, random_state=3)
    X_stf, y = oversample.fit_resample(X_train, y_train)
    X_test_stf, y_test = oversample.fit_resample(X_test, y_test)

    # Separate data into S, F, T components
    X_s, X_f, X_t = X_stf[:, 0:24], X_stf[:, 24:41], X_stf[:, 41:]
    X_test_s, X_test_f, X_test_t = X_test_stf[:, 0:24], X_test_stf[:, 24:41], X_test_stf[:, 41:]

    # Perform fusion based on the method specified
    if fusion_method == 'Concat':
        X_ft, X_st, X_sf, X_stf = perform_concat(X_s, X_f, X_t)
        X_ft_test, X_st_test, X_sf_test, X_stf_test = perform_concat(X_test_s, X_test_f, X_test_t)
    elif fusion_method == 'CBP':
        X_ft, X_st, X_sf, X_stf = perform_cpb(X_s, X_f, X_t)
        X_ft_test, X_st_test, X_sf_test, X_stf_test = perform_cpb(X_test_s, X_test_f, X_test_t)
    elif fusion_method == 'TFN':
        X_ft, X_st, X_sf, X_stf = perform_tensor_fusion(X_s, X_f, X_t)
        X_ft_test, X_st_test, X_sf_test, X_stf_test = perform_tensor_fusion(X_test_s, X_test_f, X_test_t)

    # Save processed data
    def save_data(X, filename):
        df = pd.DataFrame(X)
        df.to_csv(f'{save_path}{filename}.csv', index=False)

    # Save all datasets
    datasets = {'f_data': X_f, 't_data': X_t, 's_data': X_s,
                'ft_data': X_ft, 'st_data': X_st, 'sf_data': X_sf,'stf_data': X_stf,
                'f_test_data': X_test_f, 't_test_data': X_test_t, 's_test_data': X_test_s,
                'ft_test_data': X_ft_test, 'st_test_data': X_st_test,
                'sf_test_data': X_sf_test, 'stf_test_data': X_stf_test}
    for name, data in datasets.items():
        save_data(data, name)

    # Save default vectors
    save_data(y, 'default')
    save_data(y_test, 'default_test')




def run_process(satellite_path, financial_path, textual_path, methods):
    base_save_path = "./data/"  
    method_paths = {
        "Concat": "concat/",
        "CBP": "CBP/",
        "TFN": "TFN/"
    }

    for method in methods:
        save_path = os.path.join(base_save_path, method_paths[method])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Assuming the function process_and_save_data_general is already defined
        process_and_save_data_general(satellite_path, textual_path, financial_path, save_path, method)

satellite_path = '../data/satellite_data.xlsx'
financial_path = '../data/financial_data.xlsx'
textual_path = '../data/textual_data.xlsx'
fusion_method = ['Concat','CBP','TFN']

run_process(satellite_path, financial_path, textual_path, fusion_method)
