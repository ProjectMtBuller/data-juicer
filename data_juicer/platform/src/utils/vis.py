import plotly.graph_objects as go

def draw_sankey_diagram(source_data, target_data, value_data, labels, pos_x=[], pos_y=[]):
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels,
            x=pos_x,  # 指定节点的x坐标
            y=pos_y  # 指定节点的y坐标
        ),
        link=dict(
            source=source_data,
            target=target_data,
            value=value_data
        )
    )])
    fig.update_layout(title_text="一、数据清洗总览", title_font=dict(size=25), font_size=16)
    return fig
