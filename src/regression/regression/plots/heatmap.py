import numpy
import plotly.graph_objects as go
import plotly.express as px

def draw_heatmap(input, x, y):

    fig = px.imshow(input, text_auto=True, aspect="auto", x = x, y = y)
    fig.show()

z_mlp = [[0.0343, 0.0252, 0.0426, 0.0327, 0.0269], [0, 0.0441, 0.0625, 0.0659, 0.0464], [0, 0.0493, 0.0701, 0.0904, 0.1125]]
z_rbf = [[]]
x = [2, 3, 4, 5, 6]
y = [2, 3, 4]
draw_heatmap(z, x, y)