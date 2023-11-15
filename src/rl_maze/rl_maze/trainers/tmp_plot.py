import matplotlib
import random
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

def plot_mlp_urbf_mrbf(data_mlp, data_urbf, data_mrbf, filename):
    #random color generation in plotly
    hex_colors_dic = {}
    rgb_colors_dic = {}
    hex_colors_only = []
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_only.append(hex)
        hex_colors_dic[name] = hex
        rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)


    #calculating mean and standard deviation
    mean_mlp=np.mean(data_mlp,axis=0)
    std_mlp=np.std(data_mlp,axis=0)

    mean_urbf=np.mean(data_urbf,axis=0)
    std_urbf=np.std(data_urbf,axis=0)
    
    mean_mrbf=np.mean(data_mrbf,axis=0)
    std_mrbf=np.std(data_mrbf,axis=0)


    #draw figure
    fig = go.Figure()
    c_mlp = '#205380'
    c_urbf = '#f08d4f'
    c_mrbf = '#669900'
    fig.add_trace(go.Scatter(x=np.arange(len(data_mlp[0])), y=mean_mlp+std_mlp,
                                        mode='lines',
                                        line=dict(color=c_mlp,width =0.02),
                                        ))
    fig.add_trace(go.Scatter(x=np.arange(len(data_mlp[0])), y=mean_mlp,
                            mode='lines',
                            line=dict(color=c_mlp, width=0.8),
                            fill='tonexty',
                            name='MLP'))
    fig.add_trace(go.Scatter(x=np.arange(len(data_mlp[0])), y=mean_mlp-std_mlp,
                            mode='lines',
                            line=dict(color=c_mlp, width =0.02),
                            fill='tonexty',
                            ))

    fig.add_trace(go.Scatter(x=np.arange(len(data_urbf[0])), y=mean_urbf+std_urbf,
                                        mode='lines',
                                        line=dict(color=c_urbf,width =0.02),
                                        ))
    fig.add_trace(go.Scatter(x=np.arange(len(data_urbf[0])), y=mean_urbf,
                            mode='lines', 
                            line=dict(color=c_urbf, width=0.8),
                            fill='tonexty',
                            name='URBF'))
    fig.add_trace(go.Scatter(x=np.arange(len(data_urbf[0])), y=mean_urbf-std_urbf,
                            mode='lines',
                            line=dict(color=c_urbf, width =0.02),
                            fill='tonexty',
                            ))
    
    fig.add_trace(go.Scatter(x=np.arange(len(data_mrbf[0])), y=mean_mrbf+std_mrbf,
                                        mode='lines',
                                        line=dict(color=c_mrbf,width =0.02),
                                        ))
    fig.add_trace(go.Scatter(x=np.arange(len(data_mrbf[0])), y=mean_mrbf,
                            mode='lines', 
                            line=dict(color=c_mrbf, width=0.8),
                            fill='tonexty',
                            name='MRBF'))
    fig.add_trace(go.Scatter(x=np.arange(len(data_mrbf[0])), y=mean_mrbf-std_mrbf,
                            mode='lines',
                            line=dict(color=c_mrbf, width =0.02),
                            fill='tonexty',
                            ))

    fig.data[0].showlegend = False
    fig.data[2].showlegend = False
    fig.data[3].showlegend = False
    fig.data[5].showlegend = False
    fig.data[6].showlegend = False
    fig.data[8].showlegend = False
    #fig.data[1].showlegend = False
    #fig.data[4].showlegend = False
    #fig.data[7].showlegend = False
    
    # put the legend inside the figure at the bottom right corner
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    # increase the size of the x and y axis tick labels
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True
        ),

    )
    fig.update_layout(
        #yaxis_title="reward",
        #xaxis_title="time step (x10^3)",
        font=dict(
                family="Times New Roman",
                size=20,
                color="black"
        )

    )

    # Save figure as PDF with a rectangular shape
    width = float(210 / 25.4)  # 210mm to inches
    height = float(297 / 25.4) / 3  # 297mm divided by 3 and converted to inches
    pio.write_image(fig, filename, width=600, height=350)

