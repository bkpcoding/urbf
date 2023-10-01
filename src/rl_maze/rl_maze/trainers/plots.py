import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def plot_return_parameters(params_mlp, params_rbf, results_mlp_mean, results_mlp_std, results_rbf_mean, results_rbf_std):
    # plot the mean and std of the return for the MLP and RBF like error bars
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=params_mlp, y=results_mlp_mean, name='MLP', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=params_mlp, y=results_mlp_mean + results_mlp_std, name='MLP + std', mode='lines', line=dict(color='royalblue', width=0)))
    fig.add_trace(go.Scatter(x=params_mlp, y=results_mlp_mean - results_mlp_std, name='MLP - std', mode='lines', line=dict(color='royalblue', width=0)))
    fig.add_trace(go.Scatter(x=params_rbf, y=results_rbf_mean, name='RBF', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=params_rbf, y=results_rbf_mean + results_rbf_std, name='RBF + std', mode='lines', line=dict(color='firebrick', width=0)))
    fig.add_trace(go.Scatter(x=params_rbf, y=results_rbf_mean - results_rbf_std, name='RBF - std', mode='lines', line=dict(color='firebrick', width=0)))
    fig.update_layout(title='Mean and std of the return for the MLP and RBF like error bars', xaxis_title='Number of hidden units', yaxis_title='Return')
    fig.show()