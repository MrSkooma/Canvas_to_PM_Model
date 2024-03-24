import numpy as np
from dash import dash, html, dcc, Output, Input, State, ctx
import dash_bootstrap_components as dbc
import re


from probabilistic_model.learning.nyga_distribution import NygaDistribution
from random_events.variables import Continuous

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1('Dash'))]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="canvas",
                      figure={
                          'data': [

                          ],
                          'layout': {
                              'xaxis': {'title': ' ', 'range': [1, 10], 'fixedrange': True},
                              'yaxis': {'visible': False, 'range': [0.0001, 1.1], 'fixedrange': True},
                          }
                      }, config={"modeBarButtonsToAdd": ["eraseshape", "drawopenpath"],
                                 "modeBarButtonsToRemove": ["resetViews", "zoom2d", "zoomIn2d", "zoomOut2d",
                                                            "autoScale2d", "resetScale2d", "toImage", "lasso2d",
                                                            "select2d"],
                                 "displayModeBar": True,
                                 "displaylogo": False, "showAxisDragHandles": True, "showAxisRangeEntryBoxes": True}
                      )
        ])
    ]),
    dbc.Row([
        dbc.Col([html.Div(children="Min_X")]),
        dbc.Col(dcc.Input(id="min_x-achse", type="number", value=1)),
        dbc.Col([html.Div(children="Max_X")]),
        dbc.Col(dcc.Input(id="max_x-achse", type="number", value=10)),
        dbc.Col([html.Div(children="DPI")]),
        dbc.Col(dcc.Input(id="DPI", type="number", value=10)),
    ], className="m-3 border rounded  border-white"),
    dbc.Row([
        dbc.Button(id="plot", name="Plot", n_clicks=0)
    ]),
    dbc.Row(id="output", children=[
        dbc.Col([html.Div(children="MSQ")]),
        dbc.Col(dcc.Input(id="msq", type="number", value=10)),
        dbc.Col([html.Div(children="MLI")]),
        dbc.Col(dcc.Input(id="mli", type="number", value=0.001))
    ], className="m-3 border rounded  border-white")
])


@app.callback(
    Output("canvas", "figure"),
    Output("min_x-achse", "value"),
    Output("max_x-achse", "value"),
    Input("plot", "n_clicks"),
    Input("min_x-achse", "value"),
    Input("max_x-achse", "value"),
    State("canvas", "figure"),
    State("canvas", "relayoutData"),
    State("DPI", "value"),
    State('msq', "value"),
    State('mli', "value"),
)
def canvas(n1, min_x, max_x, fig, relayoutData, dpi, msq, mli):
    print(f"min_x: {min_x}, max_x: {max_x}, fig: {fig}")
    cb = ctx.triggered_id if not None else None
    print(cb)
    if cb is None:
        return fig, min_x, max_x
    elif cb in ["min_x-achse", "max_x-achse"]:
        if min_x is None or max_x is None:
            min_x = 1
            max_x = 10

        if min_x > max_x:
            temp = min_x
            min_x = max_x
            max_x = temp
        fig["layout"]["xaxis"]["range"] = [min_x, max_x]
        return fig, min_x, max_x,
    elif cb == "plot":
        fig["data"], new_shapes = plot(relayoutData, dpi, msq, mli, min_x, max_x)

        for i in range(len(new_shapes)):
            fig["layout"]["shapes"][i]['path'] = new_shapes[i]

        return fig, min_x, max_x
    else:
        raise Exception(f"Unknown callback: {cb}")


def plot(relayoutData, dpi, min_sample_per_quantile=10, min_likelihood_improvement=0.001, min_x=0, max_x=10):
    if dpi == 0:
        dpi = 10
    if relayoutData is None or not ('shapes' in relayoutData):
        return [], relayoutData
    data = relayoutData['shapes']
    pre_paths = [shape["path"] for shape in data]
    paths = []
    for pre_path in pre_paths:
        path_li_temp = re.split(r'(?=M|L)', pre_path)
        paths.append([st for st in path_li_temp if st])
    points = []
    relay_paths = []
    step = 1 / dpi

    for path in paths:
        pen = [min_x - 1, 0]
        relay_path = []
        x_location = min_x - step
        path_index = 0
        while path_index < len(path):
            x_location += step
            point = path[path_index]
            path_index += 1
            if point[0] == "M":
                pen = [float(p) for p in point[1:].split(',')]
                relay_path.append(('M', pen))
            elif point[0] == "L":
                line_point = [float(p) for p in point[1:].split(',')]
                relay_path.append(('L', line_point))
                if line_point[0] == pen[0]:
                    line_point[0] = line_point[0] + 0.00001

                min_point = min(pen[0], line_point[0])
                x_location = np.floor(min_point)
                max_point = max(pen[0], line_point[0])

                m = (line_point[1] - pen[1]) / (line_point[0] - pen[0])

                while max_point >= x_location:
                    if x_location > min_point:
                        points.append([x_location, m * (x_location - pen[0]) + pen[1]])
                    x_location += step
                pen = line_point
            else:
                raise ValueError("Invalid path Component")
        relay_paths.append(relay_path)

    print(len(points))
    variable = Continuous("x")
    distribution = NygaDistribution(variable, min_sample_per_quantile, min_likelihood_improvement)

    raw_weights = [point[1] for point in points]
    sum_of_weights = sum(raw_weights)
    x_values = [point[0] for point in points]
    normalized_weights = [w / sum_of_weights for w in raw_weights]
    distribution.fit(x_values, normalized_weights)
    dis_plot = distribution.plot()

    new_shapes = []
    for i in range(0, len(relay_paths)):
        relay_path = relay_paths[i]
        path_min_y = min(relay_path, key=lambda x: x[1][1])[1][1]
        path_max_y = max(relay_path, key=lambda x: x[1][1])[1][1]

        dis_exp_li = [x for x in dis_plot if x['name'] == 'Expectation']
        exp_max = dis_exp_li[0]['y'][1] if len(dis_exp_li) > 0 else path_max_y
        exp_min = dis_exp_li[0]['y'][0] if len(dis_exp_li) > 0 else path_min_y

        relayoutData_new_string = ""
        for t, point in relay_path:
            ratio = (exp_max - exp_min) * ((point[1] - path_min_y) / (path_max_y - path_min_y)) + exp_min
            relayoutData_new_string += f"{t}{point[0]},{point[1] * ratio}"
        new_shapes.append(relayoutData_new_string)

    return dis_plot, new_shapes


if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False)
