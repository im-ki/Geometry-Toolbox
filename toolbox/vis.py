import numpy as np

def draw_surface(face, vert, intensity, colormap = 'copper'):
    import matplotlib
    import plotly.graph_objects as go

    # Get the colormap
    cmap = matplotlib.colormaps[colormap]
    copper = cmap(np.linspace(0, 1, 256))[:, :3]

    colors = intensity - np.min(intensity)
    colors = np.ceil(colors / np.max(colors) * 255).astype(np.uint8)
    colors = copper[colors, :]
    mesh = go.Mesh3d(
        x=vert[:, 0],
        y=vert[:, 1],
        z=vert[:, 2],
        i=face[:, 0],
        j=face[:, 1],
        k=face[:, 2],
        vertexcolor=colors,
        intensitymode='vertex',
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        #title='Triangular Surface with Vertex Colors',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )
    fig.show()
