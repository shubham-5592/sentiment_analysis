import plotly.graph_objects as go
from numpy import pi, linspace

def plot_emotions_circle(results):
    labels = [res['label'] for res in results]
    scores = [res['score'] for res in results]
    x = {r['label']: r['score'] for r in results}
    print(x)

    # Convert to polar coordinates (circle layout)
    num_emotions = len(labels)
    angles = linspace(0, 2 * pi, num_emotions, endpoint=False).tolist()
    scores += scores[:1]  # repeat first to close the loop
    angles += angles[:1]

    # Plot
    fig = go.Figure(
        data=go.Scatterpolar(
            r=scores,
            theta=[label.capitalize() for label in labels] + [labels[0].capitalize()],
            fill='toself',
            marker=dict(color='rgba(0,123,255,0.7)')
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title="🌀 Emotion Profile"
    )
    return fig
