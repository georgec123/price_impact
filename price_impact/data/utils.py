from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

pd.options.plotting.backend = "plotly"


def plot_ticker(ticker: str, df: pd.DataFrame):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df.index, y=df.price, name="price"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df.volume, name="volume"),
        secondary_y=True,
    )
    fig.update_layout(title=ticker)
    fig.show()

