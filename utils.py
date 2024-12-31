import plotly.graph_objects as go

def plot_ternary(traj, player, n_actions, fig=None):
    if fig is None:
        fig = go.Figure()
        
    fig.add_trace(go.Scatterternary({
        'mode': 'lines',
        'a': traj[player * n_actions + 0],
        'b': traj[player * n_actions + 1],
        'c': traj[player * n_actions + 2],
        'showlegend': False
    }))

    fig.add_trace(go.Scatterternary({
        'mode': 'markers',
        'a': [traj[player * n_actions + 0, -1]],
        'b': [traj[player * n_actions + 1, -1]],
        'c': [traj[player * n_actions + 2, -1]],
        'showlegend': False
    }))
    fig.update_layout({
        'ternary': {
            'sum': 1,
            'aaxis': {'title': 'Action 1'},
            'baxis': {'title': 'Action 2'},
            'caxis': {'title': 'Action 3'},
        }})
    
    fig.update_layout(
        title="",
        font=dict(
            family="Arial",
            size=20,
            color="rgb(20, 33, 61)"
        )
    )
    return fig

def plot_3d(traj, n_actions, fig=None):
    if fig is None:
        fig = go.Figure()
        
    fig.add_trace(go.Scatter3d(x=traj[0], y=traj[n_actions], z=traj[n_actions * 2], mode='lines',
                line=dict(width=3), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[traj[0, -1]], y=[traj[n_actions, -1]], z=[traj[n_actions * 2, -1]], mode='markers',
                marker=dict(size=5), showlegend=False))
    
    fig.update_layout(scene_aspectmode='cube')

    fig.update_layout(scene_xaxis = dict(range=[0, 1]),
                    scene_yaxis = dict(range=[0, 1]),
                    scene_zaxis = dict(range=[0, 1]))

    fig.update_layout(scene = dict(
                    xaxis_title='P1',
                    yaxis_title='P2',
                    zaxis_title='P3')
    )
    
    fig.update_layout(
        title="",
        font=dict(
            family="Arial",
            size=15,
            color="rgb(20, 33, 61)"
        )
    )

    return fig