from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL, no_update, ctx, dash_table
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.colors
import chardet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer, OneHotEncoder
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, LogisticRegression, BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, make_scorer
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.class_weight import compute_sample_weight
import statsmodels.api as sm
from scipy.stats import chi2, norm, chi2_contingency
from scipy.stats import t as t_dist
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
import io, base64, webbrowser, random, os
from threading import Timer

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X has feature names, but RobustScaler was fitted without feature names")

#from sklearnex import patch_sklearn
#patch_sklearn()

app = Dash(external_stylesheets=[dbc.themes.SIMPLEX])
app.title = "線形解析ツール"

np.random.seed(0) # NumPyの乱数シードを固定
random.seed(0)    # Pythonの乱数シードを固定

# レイアウト構築
app.layout = html.Div([
    dcc.Store(id="store-upload-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-original-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-original-categories", data={"data": [], "columns": []}),
    dcc.Store(id="store-num-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-class-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-label-data", data={"data": [], "columns": []}),
    dcc.Store(id="store-rare-categories", data={"data": {}}),
    dcc.Tabs(id="tabs", value="upload-tab", children=[
        # データアップロードタブ
        dcc.Tab(
            id="upload-tab",
            value="upload-tab",
            label="①データアップロード",
            children=html.Div([
                html.Div("", style={"marginTop": "80px"}),  # タブ用のマージン
                dcc.Upload(
                    id="upload-data",
                    accept=".csv",
                    multiple=False,
                    children=html.Div([
                        html.P("CSVファイルをアップロード",
                               style={"fontSize": "22px", "fontWeight": "bold", "textAlign": "center", "marginTop": "10px"}),
                        html.P("クリックしてファイルを選択、またはここにファイルをドロップしてください",
                               style={"fontSize": "16px", "textAlign": "center", "marginTop": "10px"})
                    ]),
                ),
                html.Div(id="uploaded-data-info", style={"textAlign": "center", "padding": "50px"}),
                html.Div(
                    id="analysis-section",
                    style={"display": "none"},  # 初期状態は非表示
                    children=[
                        dbc.Accordion(
                            children=[
                                dbc.AccordionItem(
                                    title="アップロードデータ",
                                    children=[
                                        html.Div(id="data-table", style={"marginTop": "20px", "marginBottom": "20px"}),
                                        html.Div([
                                            html.Button("重複行を除外", id="exclude-duplicates", style={
                                                "margin": "5px", "padding": "5px 10px",
                                                "color": "#fff", "backgroundColor": "#e74c3c",
                                                "border": "none", "borderRadius": "5px",
                                                "cursor": "pointer"
                                            }),
                                            html.Button("表示されているデータをすべて除外", id="exclude-filtered", style={
                                                "margin": "5px", "padding": "5px 10px",
                                                "color": "#fff", "backgroundColor": "#f39c12",
                                                "border": "none", "borderRadius": "5px",
                                                "cursor": "pointer"
                                            }),
                                            html.Button("除外を全解除", id="reset-exclude", style={
                                                "margin": "5px", "padding": "5px 10px",
                                                "color": "#fff", "backgroundColor": "#3498db",
                                                "border": "none", "borderRadius": "5px",
                                                "cursor": "pointer"
                                            })
                                        ], style={"textAlign": "center", "marginBottom": "10px"}),
                                        
                                        html.H5("基礎統計量", style={"marginTop": "12px", "marginBottom": "8px"}),
                                        
                                        # 基本統計量テーブルを表示する場所
                                        html.Div(id="num-summary-table", style={"marginTop": "20px", "marginBottom": "10px"}),
                                        
                                        html.Div([
                                            html.Span("信頼区間の水準 (%)",
                                                      style={"marginRight": "10px", "color": "#34495e"}),
                                            dcc.Dropdown(
                                                id="ci-level-dropdown",
                                                options=[
                                                    {"label": "90%", "value": 0.90},
                                                    {"label": "95%", "value": 0.95},
                                                    {"label": "99%", "value": 0.99}
                                                ],
                                                value=0.95,  # デフォルトは95%
                                                clearable=False,
                                                style={"width": "150px", "margin": "auto"}
                                            )
                                        ],
                                            style={
                                                "display": "flex",
                                                "gap": "10px",              # 要素間の余白
                                                "alignItems": "center",     # 縦中央
                                                "justifyContent": "center", # 横中央（セット全体）
                                                "width": "fit-content",     # セットの幅を中身にフィット
                                                "margin": "0 auto",         # セット全体をページ中央へ
                                                "marginBottom": "30px",
                                            }),
                                        
                                        html.Div(id="cat-summary-table", style={"marginTop": "20px", "marginBottom": "20px"}),
                                    ]
                                )
                            ],
                            start_collapsed=False,
                            always_open=True,
                            style={"marginTop": "20px"}
                        ),
                        dbc.Accordion(
                            children=[
                                dbc.AccordionItem(
                                    title="データの相関",
                                    children=[
                                        # 上段：相関係数行列
                                        html.Div(
                                            id="corr-top",
                                            children=[
                                                dcc.Graph(
                                                    id="corr-matrix",
                                                    #config={"scrollZoom": True},
                                                    style={"display": "inline-block", "width": "50%", "height": "50vw", "vertical-align": "top"}
                                                ),
                                                dcc.Graph(
                                                    id="pair-plot",
                                                    #config={"scrollZoom": True},
                                                    style={"display": "inline-block", "width": "45%", "height": "45vw", "vertical-align": "top"}
                                                ),
                                            ]
                                        ),
                                        # 下段：相関比とクラメールのV
                                        html.Div(
                                            id="corr-bottom",
                                            children=[
                                                dcc.Graph(
                                                    id="eta-cramers-matrix",
                                                    style={"display": "inline-block", "width": "50%", "height": "50vw", "vertical-align": "top"}
                                                ),
                                                dcc.Graph(
                                                    id="crosstab",
                                                    style={"display": "inline-block", "width": "45%", "height": "45vw", "vertical-align": "top"}
                                                ),
                                            ],
                                            style={"marginTop": "30px"}  # 上段との間に余白
                                        )
                                    ]
                                )
                            ],
                            start_collapsed=False,
                            always_open=True,
                        ),
                        dbc.Accordion(
                            id="vif-accordion",
                            children=[
                                dbc.AccordionItem(title="多重共線性", children=[
                                        dcc.Graph(
                                            id="vif-matrix",
                                            #config={"scrollZoom": True},
                                            style={"display": "inline-block", "width": "70%", "height": "45vw"}
                                        )
                                ])
                            ],
                            start_collapsed=True,
                            always_open=True,
                            style={"marginBottom": "60px"}
                        )
                    ]
                )
            ], style={"textAlign": "center"})
        ),
        # 解析条件設定タブ
        dcc.Tab(
            id="condition-tab",
            value="condition-tab",
            label="②解析条件設定",
            style={"display": "none"},  # 初期状態は非表示
            children=[
                html.Div([
                    # 内部的なチェックボックス(常に非表示)
                    dcc.Checklist(id="explanatory-variables", options=[], value=[], style={"display": "none"}),
                    dcc.Checklist(id="target-variables", options=[], value=[], style={"display": "none"}),
                    dcc.Checklist(id="class-variables", options=[], value=[], style={"display": "none"}),
                    
                    html.Div("", style={"marginTop": "60px"}),  # タブ用のマージン
                    html.H3("解析条件を設定", style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px"}),
                    html.Div(
                        id="cat-summary-dropdown",
                        children=[
                            html.H5("エンコード設定", style={"marginTop": "12px", "marginBottom": "8px"}),
                            html.Div([
                                html.Label("レアカテゴリをまとめる:", style={"color": "#34495e", "marginLeft": "10px"}),
                                dbc.Switch(
                                    id="rare-category-switch",
                                    value=False,  # 初期状態では無効
                                    style={"marginLeft": "10px"}
                                )
                            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginTop": "20px"}),
                        ], style={"marginTop": "20px", "marginBottom": "20px"}
                    ),
                    html.Div([
                        html.Div([
                            html.Label("説明変数を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                            dcc.Checklist(
                                id="option-explanatory",
                                options=[],  # アップロード後に動的に更新
                                value=[],
                                style={"margin": "10px", "textAlign": "left"}
                            ),
                            html.Button("全て選択", id="select-all-explanatory", style={
                                "marginTop": "10px", "padding": "5px 10px",
                                "color": "#fff", "backgroundColor": "#00cc96",
                                "border": "none", "borderRadius": "5px",
                                "cursor": "pointer", "fontSize": "14px"
                            }),
                            html.Div(
                                id="impute-option",
                                children=[
                                    html.Label("欠損値補完:", style={
                                        "fontWeight": "bold",
                                        "color": "#34495e",
                                        "marginBottom": "5px",
                                        "marginTop": "10px"
                                    }),
                                    dcc.Dropdown(
                                        id="impute-mode",
                                        options=[
                                            {"label": "補完なし", "value": "none"},
                                            {"label": "中央値/最頻値", "value": "median_mode"},
                                            {"label": "線形モデル補間/「欠損」埋め", "value": "iterative_missing"},
                                        ],
                                        value="none",
                                        clearable=False,
                                        style={"flex": "1", "maxWidth": "250px", "paddingLeft": "10px"}
                                    )
                                ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}
                            )
                        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "15px",
                                  "borderRadius": "10px", "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"}),
                        html.Div([
                            html.Label("目的変数(回帰)を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                            dcc.Checklist(
                                id="option-target",
                                options=[],  # アップロード後に動的に更新
                                value=[],
                                style={"margin": "10px", "textAlign": "left"}
                            ),
                            html.Button("全て選択", id="select-all-target", style={
                                "marginTop": "10px", "padding": "5px 10px",
                                "color": "#fff", "backgroundColor": "#00cc96",
                                "border": "none", "borderRadius": "5px",
                                "cursor": "pointer", "fontSize": "14px"
                            })
                        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "15px",
                                  "borderRadius": "10px", "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"}),
                        html.Div([
                            html.Div([
                                html.Label("目的変数(分類)を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                                dcc.Checklist(
                                    id="option-class",
                                    options=[],  # アップロード後に動的に更新
                                    value=[],
                                    style={"margin": "10px", "textAlign": "left"}
                                ),
                                html.Button("全て選択", id="select-all-class", style={
                                "marginTop": "10px", "padding": "5px 10px",
                                "color": "#fff", "backgroundColor": "#00cc96",
                                "border": "none", "borderRadius": "5px",
                                "cursor": "pointer", "fontSize": "14px"
                                })
                            ],style={"width": "100%", "padding": "15px", "borderRadius": "10px",
                                      "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)", "marginBottom": "15px",
                                      "display": "block"}),
                            html.Div([
                                html.Label("ラベル変数を選択:", style={"fontWeight": "bold", "color": "#34495e", "marginBottom": "10px"}),
                                dcc.RadioItems(
                                    id="label-variables",
                                    options=[],  # アップロード後に動的に更新
                                    value=None,
                                    style={"margin": "10px", "textAlign": "left"}
                                ),
                                html.Div([
                                    html.Label("散布図の色分け:", style={"color": "#34495e", "marginLeft": "10px"}),
                                    dbc.Switch(
                                        id="color-toggle",
                                        value=False,  # 初期状態では無効
                                        style={"marginLeft": "10px"}
                                    )
                                ], style={"display": "flex", "alignItems": "center", "marginTop": "20px"})
                            ], style={"width": "100%", "padding": "15px", "borderRadius": "10px",
                              "backgroundColor": "#ecf0f1", "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)"})
                        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top",
                          "borderRadius": "10px", "backgroundColor": "#f9fbfc"})
                    ], style={"width": "90%", "margin": "0 auto", "display": "flex", "justifyContent": "space-between"}),
                     # 回帰手法を切り替えるラジオボタン
                    html.Div([
                        html.Label("解析モード:", style={"fontWeight": "bold", "color": "#34495e"}),
                        dcc.RadioItems(
                            id="regression-method",
                            options=[
                                {"label": "ElasticNet", "value": "elasticnet"},
                                {"label": "LASSO", "value": "lasso"},
                                {"label": "Ridge", "value": "ridge"},
                                {"label": "ロバスト回帰", "value": "huber"},
                                {"label": "分位点回帰", "value": "quantile"}
                            ],
                            value="elasticnet",  # デフォルトはElasticNet
                            inputStyle={"marginRight": "5px", "marginLeft": "10px"},
                            style={"display": "flex"}
                        )
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginTop": "20px"}),
                    html.Button("解析開始", id="analyze-button", n_clicks=0, style={
                        "padding": "10px 20px", "marginTop": "20px",
                        "color": "#fff", "backgroundColor": "#3498db",
                        "border": "none", "borderRadius": "5px",
                        "cursor": "pointer", "fontSize": "16px"
                    })
                ], style={"width": "80%", "margin": "0 auto", "textAlign": "center", "padding": "20px", "backgroundColor": "#f9fbfc",
                          "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
            ]
        ),
        # 解析結果タブ
        dcc.Tab(
            id="result-tab",
            value="result-tab",
            label="③解析結果",
            style={"display": "none"},
            children=html.Div([
                dcc.Tabs(
                    id="dynamic-tabs",
                    children=[],
                    style={
                        "position": "fixed", 
                        "top": "65px", 
                        "left": "50%", 
                        "width": "90%", 
                        "transform": "translateX(-50%)", 
                        "zIndex": 1000,
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                    }
            )],
            style={"textAlign": "center", "width": "90%", "margin": "0 auto", "padding": "20px",
                      "backgroundColor": "#f9f9f9", "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
        ),
        # 予測値計算タブ
        dcc.Tab(
            id="prediction-tab",
            value="prediction-tab",
            label="④予測値計算",
            style={"display": "none"},  # 初期状態は非表示
            children=[
                dcc.Store(id="store-scalers", data={}),
                html.Div("", style={"marginTop": "60px"}),  # タブ用のマージン
                html.Div([
                    # 左側のスクロール可能なスライダー部分
                    html.Div(
                        id="slider-section",
                        style={
                            "width": "35%",  # スライダー部分の幅
                            "height": "calc(100vh - 60px)",  # 画面全体の高さ
                            "overflowY": "scroll",  # 縦方向スクロールを有効化
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "20px",
                            "boxSizing": "border-box"
                        },
                        children=[
                            html.Div("", style={"marginTop": "60px"}),  # タブ用のマージン
                            html.Div(id="input-fields")  # スライダーなどを動的に配置
                        ]
                    ),
                    # 右側の予測結果部分
                    html.Div(
                        id="prediction-results",
                        style={
                            "width": "60%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "20px",
                            "borderLeft": "1px solid #ddd",
                            "position": "fixed",
                            "right": "2%",
                            "top": "50%",
                            "transform": "translateY(-50%)",
                            "boxSizing": "border-box",
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #ddd",
                            "borderRadius": "10px",
                            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"
                        },
                    )
                ], style={"display": "flex", "justifyContent": "space-between", "width": "90%", "margin": "0 auto"})
            ]
        )
    ],
    style={
        "position": "fixed",
        "top": "0",
        "width": "100%",
        "zIndex": 999,
        "backgroundColor": "white",
        "borderBottom": "1px solid #ddd",
        "textOverflow": "ellipsis",
        "whiteSpace": "nowrap"
    })
])
# コールバック: 目的変数ごとに動的にタブを生成
@app.callback(
    Output("dynamic-tabs", "children"),
    Input("analyze-button", "n_clicks"),
    [State("store-num-data", "data"),
     State("target-variables", "value"),
     State("class-variables", "value"),
     State("regression-method", "value")]
)
def update_dynamic_tabs(n_clicks, stored_data, target_variables, class_variables, regression_method):
    if not stored_data or (not target_variables and not class_variables):
        return [dcc.Tab(label="No Data", children=[html.Div("目的変数が選択されていません。")])]

    tabs = []
    total_variables = target_variables + class_variables
    
    # l1_ratioはElasticNet以外のときは非表示
    l1_ratio_style = {} if regression_method == "elasticnet" else {"display": "none"}

    for target in total_variables:

        if regression_method == "quantile":
            alpha_block = html.Div([
                html.Label(f"{target}の q (分位点：0.5で中央値) を設定:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                dcc.Slider(
                    id={"type": "alpha-slider", "index": target},
                    min=0.01,
                    max=0.99,
                    value=0.5,
                    marks={0.05: "0.05", 0.1: "0.1", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 0.9: "0.9", 0.95: "0.95"},
                    tooltip={"always_visible": True}
                )
            ])
        elif regression_method == "huber":
            alpha_block = html.Div([
                html.Label(f"{target}の t (0:最小絶対値, ∞:最小二乗) を設定:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                dcc.Slider(
                    id={"type": "alpha-slider", "index": target},
                    min=0.01,
                    max=0.99,
                    value=0.57356,
                    marks={0.01: "0(最小絶対値)", 0.33333: "0.5", 0.57356: "1.345(デフォルト)", 0.8: "4", 0.99: "∞(最小二乗)"},
                    tooltip={"always_visible": False}
                )
            ])
        else:
            alpha_block = html.Div([
                html.Label(f"{target}のalpha(正則化強度)を設定:", style={"fontWeight": "bold", "marginBottom": "10px"}),
                dcc.Slider(
                    id={"type": "alpha-slider", "index": target},
                    min=-5,
                    max=3,
                    marks={-5: "0.00001", -4: "0.0001", -3: "0.001", -2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"},
                    tooltip={"always_visible": True}
                )
            ])
            
        test_table = html.Div([
            dash_table.DataTable(
                id={"type": "test-table", "index": target},
                columns=[
                    {"name": "変数名", "id": "変数名"},
                    {"name": "回帰係数", "id": "回帰係数"},
                    {"name": "標準誤差", "id": "標準誤差"},
                    {"name": "p値", "id": "p値"},
                ],
                data=[{"変数名": "", "回帰係数": "", "標準誤差": "", "p値": ""}],
                row_selectable="multi",
                style_table={"width": "100%", "height": "36vw", "display": "inline-block", "overflowX": "auto", "overflowY": "auto"},
                style_header={"fontWeight": "bold"},
                style_cell={"textAlign": "center", "fontFamily": "Yu Gothic, Meiryo, sans-serif"},
                editable=False
            ),
            html.Div(id={"type": "wald-score", "index": target})],
            style={"display": "block" if regression_method in ["quantile", "huber"] else "none", "width": "45%", "height": "40vw"}
        )
        
        grid_search_graph = dcc.Graph(
            id={"type": "grid-search-score", "index": target},
            figure=default_fig(title="", center_text="解析中..."),
            style={"display": "none" if regression_method in ["quantile", "huber"] else "inline-block", "width": "45%", "height": "40vw"}
        )

        children_elements=[
            dcc.Store(id={"type": "store-r2-scores", "index": target}),
            dcc.Store(id={"type": "saved-cv-scores", "index": target}),
            html.Div("", style={"marginTop": "130px"}),
            alpha_block,
            html.Div([
                html.Label(f"{target}のl1_ratio(L1正則化の比率)を設定:", style={"fontWeight": "bold", "marginBottom": "10px", "marginTop": "20px"}),
                    dcc.Slider(
                        id={"type": "l1-ratio-slider", "index": target},
                        min=0,
                        max=1.0,
                        #value=0,
                        marks={0: "0: Ridge", 0.1: "0.1", 0.2: "0.2", 0.3: "0.3", 0.4: "0.4", 0.5: "0.5", 0.6: "0.6", 0.7: "0.7", 0.8: "0.8", 0.9: "0.9", 1: "1: LASSO"},
                    ),
            ], style=l1_ratio_style),
            html.Div(
                [
                    html.Div(
                        id={"type": "result-message", "index": target},
                        style={
                            "padding": "10px 20px",
                            "color": "#2c3e50",
                            "fontSize": "16px",
                            "display": "inline-block",
                            "marginTop": "15px",
                            "marginRight": "20px"
                        }
                    ),
                    html.Div(
                        [
                            html.Span("CV結果: ", style={"marginRight": "10px"}),
                            dash_table.DataTable(
                                id={"type": "cv-export-table", "index": target},
                                columns=[
                                    {"name": "行数", "id": "行数"},
                                    {"name": "fold", "id": "fold"},
                                    {"name": "y_true", "id": "y_true"},
                                    {"name": "y_pred", "id": "y_pred"},
                                ],
                                data=[],
                                export_format="csv",
                                page_action="none",
                                style_table={
                                    "height": "0px",
                                    "width": "0px",
                                    "overflow": "hidden",
                                    "border": "none",
                                },
                                style_cell={"padding": "0", "height": "0px"},
                            )
                        ],
                        id={"type": "cv-export-container", "index": target},
                        style={"display": "none", "alignItems": "center", "gap": "10px", "padding": "10px 0"}
                    )
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "padding": "10px 20px"
                }
            ),
            html.Div([
                dcc.Graph(
                    id={"type": "feature-importance-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "40vw"}
                ),
                test_table,
                grid_search_graph
            ], style={"display": "flex", "justifyContent": "center"})
        ]
        # 分類と回帰でレイアウトを切り替え
        if target in class_variables:
            # 分類の場合
            children_elements.append(
                dcc.Graph(
                    id={"type": "scatter-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    style={"display": "inline-block", "width": "50%", "height": "50vw"}
                )
            )
            children_elements.append(
                html.Div([
                    dcc.Slider(
                        id={"type": "threshold-slider", "index": target},
                        min=0.0,
                        max=1.0,
                        value=0.5,
                        marks={i: f"{i:.1f}" for i in [round(x * 0.1, 1) for x in range(0, 11)]},
                        tooltip={"always_visible": True},
                        vertical=True  # 縦型スライダーを指定
                    ),
                    html.Label("閾値", style={"fontWeight": "bold"}),
                    html.Label("", style={"fontWeight": "bold", "marginBottom": "10vw"})
                ], style={"display": "inline-block", "width": "5%", "height": "30vw"}),
            )
            children_elements.append(
                dcc.Graph(
                    id={"type": "cross-validation-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "50vw"}
                )
            )
        else:
            # 回帰の場合
            children_elements.append(
                dcc.Graph(
                    id={"type": "scatter-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "40vw"}
                )
            )
            children_elements.append(
                html.Div([
                    dcc.Slider(id={"type": "threshold-slider", "index": target},min=0, max=1, value=0),
                    ], style={"display": "none"})
            )
            children_elements.append(
                dcc.Graph(
                    id={"type": "cross-validation-plot", "index": target},
                    figure=default_fig(title="", center_text="解析中..."),
                    #config={"scrollZoom": True},
                    style={"display": "inline-block", "width": "45%", "height": "40vw"}
                )
            )
            
        tabs.append(
            dcc.Tab(id={"type": "child-tabs", "index": target}, label=f"{target}", children=
                dcc.Loading(
                    type="circle",  # プログレスリングを表示
                    id={"type": "loading", "index": target},
                    children=children_elements
                )
            )
        )

    return tabs

# Exportボタン表示制御
@app.callback(
    Output({"type": "cv-export-container", "index": ALL}, "style"),
    Input("analyze-button", "n_clicks"),
    Input({"type": "cv-export-table", "index": ALL}, "data"),
    prevent_initial_call=True
)
def toggle_export_container(n_clicks, all_data):
    styles = []
    for data in all_data:
        if data and len(data) > 0:
            styles.append({"display": "flex"})
        else:
            styles.append({"display": "none"})
    return styles

def convert_percent_columns(df):
    """
    データフレーム内の '10%' や '20.5％' などのパーセント表記文字列を検出し、
    自動的に数値（例: 0.1, 0.205）に変換して float 型の列にする関数。
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            # 各要素がパーセント表記（例: '10%', '20.5％'）か判定して数値化
            def try_parse_pct(val):
                if pd.isna(val) or val is None or val == "":
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    s = val.strip()
                    if s.endswith("%") or s.endswith("％"):
                        num_part = s[:-1].strip().replace(",", "")
                        try:
                            return float(num_part) / 100.0
                        except ValueError:
                            return val
                    else:
                        # % がついていない文字列も、純粋な数値文字列なら変換を試みる
                        try:
                            return float(s.replace(",", ""))
                        except ValueError:
                            return val
                return val

            has_pct = df[col].astype(str).str.contains(r"[%％]", na=False)
            if has_pct.any():
                converted_series = df[col].map(try_parse_pct)
                converted_numeric = pd.to_numeric(converted_series, errors="coerce")
                valid_orig = df[col].notna() & (df[col].astype(str).str.strip() != "")
                # 有効な行のうち、数値に変換できた行がすべて（欠損除外して完全に数値化できる場合）
                if valid_orig.sum() > 0 and converted_numeric[valid_orig].notna().all():
                    df[col] = converted_numeric
    return df


class CategoricalMissingNormalizer(BaseEstimator, TransformerMixin):
    """Dash/JSON経由でNoneになったカテゴリ欠損をnp.nanに統一する。"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.replace({None: np.nan, "": np.nan})
        if isinstance(X, np.ndarray):
            out = X.astype(object, copy=True)
            out[pd.isna(out)] = np.nan
            out[out == ""] = np.nan
            return out
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class RareCategoryReplacer(BaseEstimator, TransformerMixin):
    """学習データ全体で事前判定したレアカテゴリを、CV/予測データにも固定ルールで適用する。"""
    def __init__(self, enabled=False, threshold=None, protected_category=None, fixed_rare_categories=None):
        self.enabled = enabled
        self.threshold = threshold
        self.protected_category = protected_category
        self.fixed_rare_categories = fixed_rare_categories
        self.rare_category_map_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        self.rare_category_map_ = {}
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
            for col in self.columns_:
                # fixed_rare_categories が渡された場合、fold内では再判定しない。
                # 判定元は encode_data() が学習データ全体から作成したマップ。
                if self.enabled and self.fixed_rare_categories is not None:
                    self.rare_category_map_[col] = list(self.fixed_rare_categories.get(col, []))
                    continue

                values = X[col].dropna()
                threshold = self.threshold
                if threshold is None:
                    threshold = int(np.ceil(np.sqrt(len(X)) / 5))
                if self.enabled and len(values):
                    counts = values.value_counts(dropna=True)
                    rare = counts[counts <= threshold].index.tolist()
                    if self.protected_category in rare:
                        rare.remove(self.protected_category)
                    if len(rare) <= 1 or len(rare) == len(counts):
                        rare = []
                    self.rare_category_map_[col] = rare
                else:
                    self.rare_category_map_[col] = []
        else:
            self.columns_ = None
        return self

    def transform(self, X):
        rare_map = getattr(self, "rare_category_map_", {})
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col, rare_cats in rare_map.items():
                if col in X_out.columns and rare_cats:
                    X_out[col] = X_out[col].mask(X_out[col].isin(rare_cats), "その他")
            return X_out
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """fold内で観測したカテゴリだけでOHE。基準カテゴリが存在しないfoldではdropしない。"""
    def __init__(self, col_name=None, base_category=None, all_categories=None):
        self.col_name = col_name
        self.base_category = base_category
        # all_categories は後方互換用。CVで全データ由来のカテゴリ一覧は使わない。
        self.all_categories = all_categories
        self.encoder_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        col_name = self.col_name or "cat"
        probe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        probe.fit(X)
        observed = list(probe.categories_[0])
        drop = [self.base_category] if self.base_category in observed else None
        self.encoder_ = OneHotEncoder(
            categories=[observed], drop=drop, handle_unknown="ignore", sparse_output=False
        )
        self.encoder_.fit(X)
        cleaned_names = []
        for fn in self.encoder_.get_feature_names_out([col_name]):
            suffix = fn[len(col_name) + 1:]
            cleaned_names.append(f"{col_name}[{suffix}]")
        self.feature_names_out_ = np.asarray(cleaned_names, dtype=object)
        return self

    def transform(self, X):
        return self.encoder_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class NumericPreprocessor:
    def __init__(self, impute_method: str = "none", scale: bool = True, base_categories=None, rare_category_map=None, all_categories=None, rare_category_enabled=False):
        self.impute_method = impute_method
        self.scale = scale
        self.base_categories = base_categories or {}
        self.rare_category_map = rare_category_map or {}
        self.all_categories = {}  # CVでは全データ由来のカテゴリ一覧を使わない
        self.rare_category_enabled = rare_category_enabled
        self.column_transformer = None
        self.scaler = None
        self.is_df = False
        self.columns = None
        self.index = None
        self.feature_names_in_ = None
        self.transformed_feature_names_ = None
        self.numeric_idx_ = []
        self.dummy_idx_ = []
        self.n_cols_ = 0
        self.cat_cols_ = []
        self.dummy_cols_ = []

    def fit(self, X, y=None):
        if X is None or (hasattr(X, 'shape') and X.shape[0] == 0):
            return self

        self.is_df = isinstance(X, pd.DataFrame)
        if self.is_df:
            self.columns = list(X.columns)
            self.index = X.index
            self.feature_names_in_ = list(X.columns)
            X_df = X.copy()
        else:
            self.columns = None
            self.index = None
            n_cols = X.shape[1] if len(X.shape) > 1 else 0
            self.feature_names_in_ = [f"col_{i}" for i in range(n_cols)]
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        cat_cols = []
        dummy_cols = []
        num_cols = []
        for c in self.feature_names_in_:
            if c in self.base_categories:
                cat_cols.append(c)
            elif "[" in str(c) and str(c).endswith("]"):
                dummy_cols.append(c)
            else:
                if not pd.api.types.is_numeric_dtype(X_df[c]):
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

        # カテゴリ列は元データが整数dtype（例: 1, 2, 3）でも、
        # iterative_missing では文字列「欠損」を補完値として使う。
        # pandasの整数dtypeのままだと SimpleImputer が文字列を代入できず、
        # fit時または予測transform時に dtype cast エラーになるため、
        # fit/transformで同じ列をobject dtypeにそろえる。
        self.cat_cols_ = list(cat_cols)
        self.dummy_cols_ = list(dummy_cols)
        for c in self.cat_cols_:
            X_df[c] = X_df[c].astype(object)
        # One-hot列は欠損補完の対象としてfloatに統一する。
        # int64列にSimpleImputer(fill_value=0)がfloat値を渡す環境で発生する
        # 「filling value float64 cannot be cast to int64」を防ぐ。
        for c in dummy_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").astype(float)

        if self.impute_method == "none" and num_cols and X_df[num_cols].isna().any().any():
            missing_cols = [c for c in num_cols if X_df[c].isna().any()]
            raise ValueError(
                "補完なしでは数値説明変数の欠損値を扱えません。欠損のある列: " + ", ".join(map(str, missing_cols))
                + "。欠損値補完を選択するか、欠損行を除外してください。"
            )

        transformers = []

        if num_cols:
            if self.impute_method in ["median", "median_mode"]:
                num_imp = SimpleImputer(strategy="median")
            elif self.impute_method in ["linear", "iterative_missing"]:
                num_imp = IterativeImputer(estimator=BayesianRidge(), random_state=0)
            elif self.impute_method == "zero":
                num_imp = SimpleImputer(strategy="constant", fill_value=0)
            elif self.impute_method == "non-linear":
                num_imp = IterativeImputer(
                    estimator=ExtraTreesRegressor(
                        n_estimators=30, max_depth=5, random_state=0, n_jobs=-1
                    ),
                    max_iter=2,
                    initial_strategy="median",
                    random_state=0
                )
            else:
                num_imp = FunctionTransformer()

            num_steps = [("imputer", num_imp)]
            if self.scale:
                num_steps.append(("scaler", RobustScaler()))
            transformers.append(("numeric", Pipeline(num_steps), num_cols))

        if dummy_cols:
            if self.impute_method in ["median_mode", "mode"]:
                dummy_imp = SimpleImputer(strategy="most_frequent")
            elif self.impute_method in ["iterative_missing", "missing"]:
                dummy_imp = SimpleImputer(strategy="constant", fill_value=0)
            elif self.impute_method == "zero":
                dummy_imp = SimpleImputer(strategy="constant", fill_value=0)
            else:
                dummy_imp = FunctionTransformer()
            transformers.append(("dummy", Pipeline([("imputer", dummy_imp)]), dummy_cols))

        for c in cat_cols:
            if self.impute_method in ["iterative_missing", "missing"]:
                cat_imp = SimpleImputer(strategy="constant", fill_value="欠損")
            elif self.impute_method in ["median_mode", "mode"]:
                cat_imp = SimpleImputer(strategy="most_frequent")
            elif self.impute_method == "zero":
                cat_imp = SimpleImputer(strategy="constant", fill_value="0")
            else:
                cat_imp = FunctionTransformer()

            base_cat = self.base_categories.get(c, None)
            all_cats = self.all_categories.get(c, None)
            pipe = Pipeline([
                ("normalize_missing", CategoricalMissingNormalizer()),
                ("replacer", RareCategoryReplacer(
                    enabled=self.rare_category_enabled,
                    protected_category=self.base_categories.get(c),
                    fixed_rare_categories=self.rare_category_map if self.rare_category_enabled else None
                )),
                ("imputer", cat_imp),
                ("ohe", CustomOneHotEncoder(col_name=c, base_category=base_cat, all_categories=None))
            ])
            transformers.append((f"cat_{c}", pipe, [c]))

        if transformers:
            self.column_transformer = ColumnTransformer(transformers=transformers)
            self.column_transformer.fit(X_df)
            final_names = []
            for name, trans, cols in self.column_transformer.transformers_:
                if name == "remainder":
                    continue
                try:
                    final_names.extend(trans.get_feature_names_out(cols))
                except Exception:
                    # Numeric列は元の列名を維持。カテゴリ列はOHEの名前を使用。
                    if name == "numeric":
                        final_names.extend(list(cols))
                    elif name == "dummy":
                        final_names.extend(list(cols))
                    else:
                        try:
                            final_names.extend(trans.named_steps["ohe"].get_feature_names_out(cols))
                        except Exception:
                            final_names.extend(list(cols))
            self.transformed_feature_names_ = list(final_names)
        else:
            self.column_transformer = None
            self.transformed_feature_names_ = list(self.feature_names_in_)

        class ProxyScaler:
            def __init__(self, center, scale):
                self.center_ = np.array(center)
                self.scale_ = np.array(scale)

        num_scaler = None
        if num_cols and self.scale and self.column_transformer is not None:
            num_pipe = self.column_transformer.named_transformers_["numeric"]
            if "scaler" in num_pipe.named_steps:
                num_scaler = num_pipe.named_steps["scaler"]

        full_scale = [1.0] * len(self.transformed_feature_names_)
        full_center = [0.0] * len(self.transformed_feature_names_)
        for k, col_name in enumerate(num_cols):
            if num_scaler is not None and col_name in self.transformed_feature_names_:
                idx = self.transformed_feature_names_.index(col_name)
                full_scale[idx] = num_scaler.scale_[k]
                full_center[idx] = num_scaler.center_[k]

        self.scaler = ProxyScaler(full_center, full_scale)
        return self

    def transform(self, X):
        if X is None or (hasattr(X, 'shape') and X.shape[0] == 0):
            return X

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            cols = list(X.columns)
            idx = X.index
            if self.feature_names_in_ is not None and all(c in X.columns for c in self.feature_names_in_):
                X_df = X[self.feature_names_in_]
            elif (
                self.feature_names_in_ is not None
                and len(X.columns) == len(self.feature_names_in_)
                and all(str(c).startswith("col_") for c in self.feature_names_in_)
            ):
                # 学習時にNumPy配列が渡されて col_0, col_1, ... と命名された旧経路との互換性。
                # 列数が一致する場合に限り、予測入力を学習時の列順として扱う。
                X_df = X.copy()
                X_df.columns = self.feature_names_in_
            else:
                missing = set(self.feature_names_in_ or []) - set(X.columns)
                if missing:
                    raise ValueError(f"予測データに学習時の説明変数がありません: {sorted(missing)}")
                X_df = X
        else:
            cols = self.columns
            idx = None
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        # fit時と同様にカテゴリ列のdtypeを統一する。特に整数カテゴリ列へ
        # fill_value="欠損" を適用するケースで、fit/transform間のdtype不一致を防ぐ。
        X_df = X_df.copy()
        for c in getattr(self, "cat_cols_", []):
            if c in X_df.columns:
                X_df[c] = X_df[c].astype(object)
        # fit()内でdummy列をfloatにしても、transform()には元のDataFrameが
        # 再び渡されるため、ここでも同じdtype変換を行う必要がある。
        # これをしないと SimpleImputer が学習時のfloatと入力時のint64の
        # dtype不一致で失敗する。
        for c in getattr(self, "dummy_cols_", []):
            if c in X_df.columns:
                X_df[c] = pd.to_numeric(X_df[c], errors="coerce").astype(float)

        if self.column_transformer is None:
            X_res = X_df.values
        else:
            X_res = self.column_transformer.transform(X_df)

        if is_df:
            out_cols = self.transformed_feature_names_ if self.transformed_feature_names_ is not None else cols
            return pd.DataFrame(X_res, columns=out_cols, index=idx)
        else:
            return X_res

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def build_cv_feature_frame(raw_data, encoded_data, explanatory_variables):
    """UI上の選択済み特徴量から、CV用の生カテゴリ列を復元したDataFrameを作る。

    数値特徴量はそのまま、カテゴリのダミー列群は元のカテゴリ列1本に戻す。
    これによりCV fold内でカテゴリ補完・レアカテゴリ統合・OHEをfitできる。
    """
    raw_df = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame(raw_data.get("data", []), columns=raw_data.get("columns"))
    encoded_df = encoded_data if isinstance(encoded_data, pd.DataFrame) else pd.DataFrame(encoded_data.get("data", []), columns=encoded_data.get("columns"))
    selected = []
    for var in (explanatory_variables or []):
        if var in raw_df.columns:
            selected.append(var)
            continue
        if "[" in str(var) and str(var).endswith("]"):
            base = str(var).split("[", 1)[0]
            if base in raw_df.columns and not pd.api.types.is_numeric_dtype(raw_df[base]):
                selected.append(base)
                continue
        if var in encoded_df.columns:
            selected.append(var)
    # 同じカテゴリ列のダミーが複数選ばれていても元列は一度だけ含める
    selected = list(dict.fromkeys(selected))
    if not selected:
        return pd.DataFrame(index=encoded_df.index)
    out = pd.DataFrame(index=encoded_df.index)
    for col in selected:
        if col in raw_df.columns and col not in encoded_df.columns:
            out[col] = raw_df[col]
        elif col in raw_df.columns and not pd.api.types.is_numeric_dtype(raw_df[col]):
            out[col] = raw_df[col]
        elif col in encoded_df.columns:
            out[col] = encoded_df[col]
        else:
            out[col] = raw_df[col]
    return out


def transformed_feature_frame(preprocessor, X, index):
    """fit済み前処理器がモデルへ渡す特徴量を、名前付きDataFrameで返す。"""
    transformed = preprocessor.transform(X)
    if isinstance(transformed, pd.DataFrame):
        result = transformed.copy()
        result.index = index
        return result
    names = getattr(preprocessor, "transformed_feature_names_", None)
    if names is None or len(names) != np.asarray(transformed).shape[1]:
        names = [f"feature_{i + 1}" for i in range(np.asarray(transformed).shape[1])]
    return pd.DataFrame(np.asarray(transformed), columns=list(names), index=index)

class ModelPipeline:
    def __init__(self, preprocessor, estimator, is_statsmodels=False, statsmodels_type=None, fit_params=None):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.is_statsmodels = is_statsmodels
        self.statsmodels_type = statsmodels_type  # "huber", "quantile", "firth"
        self.fit_params = fit_params or {}
        self.fitted_estimator_ = None

    def _named_transformed(self, X_trans):
        """変換後特徴量に前処理器が保持する実際の特徴量名を付ける。"""
        names = getattr(self.preprocessor, "transformed_feature_names_", None)
        if isinstance(X_trans, pd.DataFrame):
            result = X_trans.copy()
            if names is not None and len(names) == result.shape[1]:
                result.columns = list(names)
            return result
        values = np.asarray(X_trans)
        if names is None or len(names) != values.shape[1]:
            names = [f"特徴量{i + 1}" for i in range(values.shape[1])]
        return pd.DataFrame(values, columns=list(names))

    def fit(self, X, y):
        X_trans = self.preprocessor.fit_transform(X)
        X_named = self._named_transformed(X_trans)
        if self.is_statsmodels:
            import statsmodels.api as sm
            X_df = X_named
            X_with_const = sm.add_constant(X_df, has_constant='add')
            if self.statsmodels_type == "huber":
                t = self.fit_params.get("t", 1.345)
                model = sm.RLM(y, X_with_const, M=sm.robust.norms.HuberT(t=t))
                self.fitted_estimator_ = model.fit()
            elif self.statsmodels_type == "quantile":
                q = self.fit_params.get("q", 0.5)
                model = sm.QuantReg(y, X_with_const)
                self.fitted_estimator_ = model.fit(q=q, method="bootstrap")
            elif self.statsmodels_type == "firth":
                self.fitted_estimator_ = firth_logistic_regression(X_named, y, scale=False, balanced=True)
        else:
            self.estimator.fit(X_named, y)
            self.fitted_estimator_ = self.estimator
        return self

    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        X_named = self._named_transformed(X_trans)
        if self.is_statsmodels:
            import statsmodels.api as sm
            if self.statsmodels_type in ["huber", "quantile"]:
                X_df = X_named
                X_with_const = sm.add_constant(X_df, has_constant='add')
                return self.fitted_estimator_.predict(X_with_const)
            elif self.statsmodels_type == "firth":
                beta_hat = self.fitted_estimator_["coefficients"]
                input_with_intercept = np.hstack([np.ones((X_named.shape[0], 1)), X_named.to_numpy()])
                eta = np.dot(input_with_intercept, beta_hat)
                prob = expit(eta)
                return prob
        else:
            return self.fitted_estimator_.predict(X_named)

    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        X_named = self._named_transformed(X_trans)
        if self.is_statsmodels and self.statsmodels_type == "firth":
            return self.predict(X)
        else:
            return self.fitted_estimator_.predict_proba(X_named)

def default_fig(title="", center_text=""):
    fig = go.Figure()
    fig.update_layout(
        title={"text": title},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        dragmode=False,
        shapes=[{"type": "rect", "xref": "paper", "yref": "paper", "x0": 0, "y0": 0, "x1": 1, "y1": 1, "line": {"color": "lightgray", "dash": "dash"}}],
        annotations=[{"text": center_text, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 16, "color": "black"}, "align": "center"}]
    )
    return fig

def generate_fold_colors(n_folds):
    color_scale = plotly.colors.sequential.Viridis  # Viridisカラースケールを使用
    colors = [color_scale[i * len(color_scale) // n_folds] for i in range(n_folds)]
    return colors

def scatter_layout(fig, x, y, title):
    x = np.array(x)
    y = np.array(y)

    # 回帰直線データを計算
    regression_model = LinearRegression()
    regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    y_line = regression_model.predict(x.reshape(-1, 1))
    r2 = concordance_corrcoef(x, y)

    # 散布図
    fig.add_trace(go.Scatter(x=x, y=y_line.flatten(), mode="lines", name="回帰直線", line=dict(color="red")))
    padding = (max(x) - min(x)) * 0.05  # 範囲の5%を余白として追加
    min_val = min(x.min(), y.min()) - padding
    max_val = max(x.max(), y.max()) + padding

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="対角線",
        line=dict(color="rgba(128, 128, 128, 0.5)", dash="dash")
    ))

    fig.update_layout(
        title=f"{title} [適合度: {r2:.4f}]",
        xaxis=dict(title="実測値", range=[min_val, max_val], scaleanchor="y", constrain="domain"),
        yaxis=dict(title="予測値", range=[min_val, max_val], constrain="domain"),
        showlegend=True,
        #dragmode="pan"
    )
    return fig

def compute_hc3_cov(result):
    resid = result.resid
    X = result.model.exog

    # (X'X)^-1 の安定計算
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(X.T @ X)

    # Hat行列とHC3補正
    hat_matrix = X @ xtx_inv @ X.T
    h = np.diag(hat_matrix)
    s = (resid / (1 - h))**2
    S = np.diag(s)

    # HC3共分散行列
    cov_hc3 = xtx_inv @ (X.T @ S @ X) @ xtx_inv
    return cov_hc3

def firth_logistic_regression(X, y, scale=True, balanced=True):
    # 入力をDataFrame化
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

    # 標準化
    if scale:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # デザイン行列（切片追加）
    X_design = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    feature_names = ["(切片)"] + feature_names
    n_params = X_design.shape[1]

    # クラス重み（balanced）
    if balanced:
        sample_weight = compute_sample_weight(class_weight='balanced', y=y)
    else:
        sample_weight = np.ones_like(y)

    # 負の修正対数尤度
    def neg_penalized_loglik(beta):
        eta = X_design @ beta
        mu = np.clip(expit(eta), 1e-8, 1 - 1e-8)
        loglik = np.sum(sample_weight * (y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        w = mu * (1 - mu)
        Fisher = X_design.T @ (X_design * (w * sample_weight)[:, None]) + np.eye(n_params) * 1e-3
        sign, logdet = np.linalg.slogdet(Fisher)
        penalty = 0.5 * logdet if sign > 0 else 0.0
        return -(loglik + penalty)

    # 初期値（ゼロベクトル）
    init_beta = np.zeros(n_params)

    # 最適化（L-BFGS-B）
    result = minimize(
        neg_penalized_loglik,
        init_beta,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-9}
    )

    if not result.success:
        raise RuntimeError(f"最適化に失敗しました: {result.message}")

    beta_hat = result.x
    eta = X_design @ beta_hat
    mu = np.clip(expit(eta), 1e-9, 1 - 1e-9)

    # 共分散行列の計算
    w = mu * (1 - mu)
    Fisher = X_design.T @ (X_design * w[:, None]) + np.eye(n_params) * 1e-6
    try:
        c, lower = cho_factor(Fisher)
        cov_matrix = cho_solve((c, lower), np.eye(n_params))
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(Fisher)

    se = np.sqrt(np.diag(cov_matrix))
    z_scores = beta_hat / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    result_df = pd.DataFrame({
        "変数名": feature_names,
        "回帰係数": np.round(beta_hat, 4),
        "標準誤差": np.round(se, 4),
        "p値": np.round(p_values, 4)
    })

    return {
        "summary": result_df,
        "predict_proba": mu,
        "predict": (mu >= 0.5).astype(int),
        "coefficients": beta_hat,
        "standard_errors": se,
        "p_values": p_values
    }

def concordance_corrcoef(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def huber_loss(y_true, y_pred, delta=1.345):
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    squared = 0.5 * error**2
    linear = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small, squared, linear).mean()

huber_scorer = make_scorer(huber_loss, greater_is_better=False)

# コールバック: タブの表示管理
@app.callback(
    [Output("condition-tab", "style"),
     Output("result-tab", "style"),
     Output("prediction-tab", "style"),
     Output("tabs", "value")],
    [Input("upload-data", "contents"),
     Input("dynamic-tabs", "children")],
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def manage_tabs(upload, dinamic_tab, contents):
    # トリガーされたイベントを判定
    triggered_id = ctx.triggered_id

    if triggered_id == "upload-data":
        # データアップロード時に結果タブを非表示
        return {"display": "block"}, {"display": "none"}, {"display": "none"}, "upload-tab"

    if triggered_id == "dynamic-tabs" and contents:
        # 解析ボタン押下時に結果タブを表示
        return {"display": "block"}, {"display": "block"}, {"display": "block"}, "result-tab"

    # 初期状態（どのトリガーも発生していない場合）
    return {"display": "none"}, {"display": "none"}, {"display": "none"}, "upload-tab"


# コールバック: ファイルアップロード時の処理
@app.callback(
    [Output("upload-data", "style"),
     Output("uploaded-data-info", "children"),
     Output("store-upload-data", "data"),
     Output("data-table", "children"),
     Output("analysis-section", "style")],
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def process_upload(contents, filename):
    upload_style = {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "flexDirection": "column",
        "height": "150px",
        "width": "50%",
        "margin": "0 auto",
        "border": "2px dashed #cccccc",
        "borderRadius": "10px",
        "backgroundColor": "#f9f9f9",
        "textAlign": "center",
        "cursor": "pointer"
    }

    # --- ファイル未アップロード時 ---
    if not contents or not filename:
        return (
            upload_style,
            "ファイルがアップロードされていません。",
            None,
            html.Div("", style={"textAlign": "center", "color": "#888", "padding": "20px"}),
            {"display": "none"}
        )

    # --- ファイルアップロード時 ---
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # エンコード形式を判定
        result = chardet.detect(decoded)
        encoding = result["encoding"] if result["encoding"] else "utf-8"

        # データを読み込む
        try:
            data = pd.read_csv(io.StringIO(decoded.decode(encoding)))
        except:
            data = pd.read_csv(io.StringIO(decoded.decode("cp932", errors="replace")))

        # 不要なUnnamed列を除去
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

        # 「%」付き表記を数値に変換（例: 10% -> 0.1）
        data = convert_percent_columns(data)

        # NumPy 型 → Python ネイティブ型へ変換
        data = data.where(pd.notnull(data), None)
        data = data.astype(object)
        data = data.map(lambda x: x.item() if hasattr(x, "item") else x)

        # 基本情報
        n_rows, n_cols = data.shape
        n_num = data.select_dtypes(include=[np.number]).shape[1]
        n_cat = data.select_dtypes(exclude=[np.number]).shape[1]
        info_message = [
            f"アップロードされたファイル: {filename}", html.Br(),
            f"{n_rows}行×{n_cols}列  (数値列: {n_num}, カテゴリ列: {n_cat})"
        ]

        # 保存用データ（元列名のまま）
        save_data = data.copy()
        save_data.insert(0, "行番号", range(1, len(save_data) + 1))
        save_data["除外"] = False

        # 表示用データ（列名の . を ． に変換）
        grid_data = save_data.copy()
        grid_data.columns = [
            c.replace(".", "．") if isinstance(c, str) else c
            for c in grid_data.columns
        ]

        table_data = grid_data.to_dict("records")

        # columnDefs（表示用の列名に合わせる）
        column_defs = [
            {
                "headerName": "除外",
                "field": "除外",
                "cellEditor": "agCheckboxCellEditor",
                "editable": True,
                "pinned": "left",
                "width": 70,
                "cellStyle": {"textAlign": "center"}
            },
            {
                "headerName": "行番号",
                "field": "行番号",
                "sortable": True,
                "filter": True,
                "pinned": "left",
                "cellStyle": {
                    "backgroundColor": "#f0f0f0",
                    "fontWeight": "bold",
                    "textAlign": "right"
                },
                "width": 70
            }
        ] + [
            {
                "headerName": col.replace(".", "．") if isinstance(col, str) else col,
                "field": col.replace(".", "．") if isinstance(col, str) else col,
                "sortable": True,
                "filter": True,
                "width": 45 + len(str(col)) * 12
            }
            for col in data.columns
        ]

        data_table = dag.AgGrid(
            id="ag-grid",
            columnDefs=column_defs,
            rowData=table_data,
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions={
                "suppressRowClickSelection": True,
                "undoRedoCellEditing": True,
                "singleClickEdit": True,
                "stopEditingWhenCellsLoseFocus": True,
            },
            style={
                "height": "500px",
                "width": "95%",
                "margin": "0 auto",
                "border": "1px solid #ddd"
            },
        )

        return (
            {"display": "none"},
            info_message,
            {"data": save_data.to_dict("records"), "columns": list(save_data.columns)},
            data_table,
            {"display": "block"}
        )

    except Exception as e:
        return (
            upload_style,
            f"エラー: {str(e)}",
            None,
            html.Div(
                "エラーが発生しました。データが処理できません。",
                style={"textAlign": "center", "color": "red", "padding": "20px"}
            ),
            {"display": "none"}
        )


@app.callback(
    Output("ag-grid", "rowData", allow_duplicate=True),
    [Input("exclude-duplicates", "n_clicks"),
     Input("exclude-filtered", "n_clicks"),
     Input("reset-exclude", "n_clicks")],
    [State("ag-grid", "rowData"),
     State("ag-grid", "virtualRowData")],
    prevent_initial_call=True
)
def update_exclude(dup_clicks, filter_clicks, reset_clicks, row_data, virtual_data):
    if not ctx.triggered or row_data is None:
        return no_update

    triggered_id = ctx.triggered_id
    df = pd.DataFrame(row_data)

    if "除外" not in df.columns:
        df["除外"] = False

    if triggered_id == "exclude-duplicates":
        dup_mask = df.duplicated(
            subset=df.columns.difference(["除外", "行番号"]),
            keep="last"
        )
        df.loc[dup_mask, "除外"] = True

    elif triggered_id == "exclude-filtered" and virtual_data:
        v_indices = pd.DataFrame(virtual_data)["行番号"].values
        df.loc[df["行番号"].isin(v_indices), "除外"] = True

    elif triggered_id == "reset-exclude":
        df["除外"] = False

    return df.to_dict("records")


@app.callback(
    Output("store-original-data", "data"),
    [Input("store-upload-data", "data"),
     Input("ag-grid", "rowData"),
     Input("ag-grid", "cellValueChanged")],
    prevent_initial_call=True
)
def update_store(upload_data, rowData, cellChanged):
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return no_update

    # アップロード直後
    if triggered_id == "store-upload-data" and upload_data is not None:
        df = pd.DataFrame(upload_data["data"], columns=upload_data["columns"])

    # チェックボックス変更時
    elif triggered_id == "ag-grid" and cellChanged:
        df = pd.DataFrame(rowData)
        changed_row = cellChanged[0]["data"]
        changed_id = changed_row["行番号"]
        exclude_flag = changed_row["除外"]
        df.loc[df["行番号"] == changed_id, "除外"] = exclude_flag

    # rowData 更新時（ボタン操作など）
    elif triggered_id == "ag-grid" and rowData is not None:
        df = pd.DataFrame(rowData)

    else:
        return no_update

    if "除外" in df.columns:
        df_filtered = df[df["除外"] == False].drop(columns=["除外"], errors="ignore")
    else:
        df_filtered = df

    # 表示用の全角ドットを元に戻して保存
    df_filtered.columns = [
        c.replace("．", ".") if isinstance(c, str) else c
        for c in df_filtered.columns
    ]

    return {
        "data": df_filtered.to_dict("records"),
        "columns": list(df_filtered.columns)
    }

@app.callback(
    [Output("num-summary-table", "children"),
     Output("cat-summary-table", "children"),
     Output("cat-summary-dropdown", "children")],
    [Input("store-original-data", "data"),
     Input("ci-level-dropdown", "value")]
)
def update_summary_table(stored_data, ci_level):
    if stored_data is None:
        return "データがありません", "データがありません", ""

    df = pd.DataFrame(stored_data["data"], columns=stored_data["columns"])
    # 数値データのみ抽出
    df_numeric = df.select_dtypes(include=[np.number])
    df_category = df.select_dtypes(exclude=[np.number])

    # --- 数値データのサマリー ---
    def summary_stats(df, ci_level=0.95):
        stats = []
        n_total = len(df)
        for col in df.columns:
            if col == "行番号":
                continue
            n = df[col].count()
            mean = df[col].mean()
            median = df[col].median()
            max_val = df[col].max()
            min_val = df[col].min()
            n_missing = df[col].isna().sum()
            missing_pct = (n_missing / n_total) * 100
            std = df[col].std(ddof=1)
            se = std / np.sqrt(n)
    
            # 信頼区間
            alpha = 1 - ci_level
            t_crit = t_dist.ppf(1 - alpha/2, df=n-1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
    
            stats.append({
                "変数": col,
                "n": n,
                "平均値": round(mean, 3),
                "標準偏差": round(std, 3),
                "最小値": round(min_val, 3),
                "中央値": round(median, 3),
                "最大値": round(max_val, 3),
                "欠損数": f"{n_missing} ({missing_pct:.1f}%)" ,
                f"母平均の{int(ci_level*100)}%信頼区間": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            })
        return pd.DataFrame(stats)
    
    # 基本統計量を計算
    if df_numeric.shape[1] > 0:
        summary_df = summary_stats(df_numeric, ci_level=ci_level)
        num_summary_table = dash_table.DataTable(
            data=summary_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in summary_df.columns],
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": "bold"},
            style_cell={"textAlign": "center", "fontFamily": "Yu Gothic, Meiryo, sans-serif"}
        )
    else:
        num_summary_table = ""  # 数値データがないときは表示しない

    # --- カテゴリデータのサマリー ---
    def cat_summary_stats(df):
        stats = []
        n_total = len(df)
        for col in df.columns:
            n = df[col].count()
            n_missing = df[col].isna().sum()
            missing_pct = (n_missing / n_total) * 100
            n_unique = df[col].nunique(dropna=True)
            unique_ratio = (n_unique / n) * 100
            value_counts = df[col].value_counts(dropna=True)  # 出現回数のカウント 
            rare_threshold = int(np.ceil(np.sqrt(n_total) / 5))
            rare_rows = value_counts[value_counts <= rare_threshold].sum()  # レアカテゴリの行数 
            rare_ratio = (rare_rows / n) * 100
    
            if n > 0:
                mode_val = df[col].mode().iloc[0]
                mode_count = (df[col] == mode_val).sum()
                mode_pct = (mode_count / n) * 100
                mode_display = f"{mode_val} ({mode_count}件, {mode_pct:.1f}%)"
            else:
                mode_display = None
    
            stats.append({
                "変数": col,
                "n": n,
                "カテゴリ数": n_unique,
                "最頻値": mode_display,
                "ユニーク率": f"{unique_ratio:.1f}%",
                f"レアカテゴリ率(n≦{rare_threshold})": f"{rare_ratio:.1f}%",
                "欠損数": f"{n_missing} ({missing_pct:.1f}%)"
            })
        return pd.DataFrame(stats)

    if df_category.shape[1] > 0:
        cat_summary_df = cat_summary_stats(df_category)
        cat_summary_df.reset_index(drop=True, inplace=True)  # 行番号を連番に
        
        # --- カテゴリサマリー表 (DataTable) ---
        cat_summary_table = dash_table.DataTable(
            data=cat_summary_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in cat_summary_df.columns],
            style_table={"overflowX": "auto"},
            style_header={"fontWeight": "bold"},
            style_cell={"textAlign": "center", "fontFamily": "Yu Gothic, Meiryo, sans-serif"}
        )
        
        # --- 行ごとの Dropdown を作成 ---
        dropdown_rows = []
        for idx, row in cat_summary_df.iterrows():
            var_name = row["変数"]
            categories = df_category[var_name].dropna().unique()
            n_unique = len(categories)
            n = df_category[var_name].count()
            unique_ratio = (n_unique / n) * 100 if n > 0 else 100
            
            # ユニーク率が70%以上ならスキップ（エンコード対象外）
            if unique_ratio >= 70:
                dropdown_rows.append(
                    dcc.Dropdown(
                        id={"type": "dynamic-dropdown", "col": var_name},
                        options=[{"label": "しない", "value": "しない"}],
                        value="しない",
                        style={"display": "none"}  # ← 非表示にする
                    )
                )
                continue
            # 欠損を除いて "O"/"X" だけならスキップ（エンコード対象外）
            if set(categories) <= {"O", "X"}:
                dropdown_rows.append(
                    dcc.Dropdown(
                        id={"type": "dynamic-dropdown", "col": var_name},
                        options=[{"label": "しない", "value": "しない"}],
                        value="しない",
                        style={"display": "none"}
                    )
                )
                continue    
            
            options = [
                {"label": "しない", "value": "しない"},
                {"label": "目的変数", "value": "目的変数"}
            ]
            # 基準:○○ を追加
            options += [{"label": f"説明変数(基準:{cat})", "value": f"説明変数(基準:{cat})"} for cat in categories]
        
            dropdown_rows.append(
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "4px"},
                    children=[
                        html.Div(row["変数"], style={"flex": "0 0 20%", "textAlign": "right", "paddingRight": "10px"}),
                        dcc.Dropdown(
                            id={"type": "dynamic-dropdown", "col": var_name},
                            options=options,
                            value="しない",
                            clearable=False,
                            style={"flex": "0 0 40%"}
                        )
                    ]
                )
            )
        
        # ドロップダウンのコンテナにタイトルを追加
        visible_dropdowns = [
            d for d in dropdown_rows
            if not (isinstance(d, dcc.Dropdown) and d.style.get("display") == "none")
        ]
        if visible_dropdowns:  # 1つ以上ドロップダウンがあるときだけ表示
            cat_summary_dropdown = html.Div(
                children=[
                    html.H5("エンコード設定", style={"marginTop": "12px", "marginBottom": "8px"}),
                    html.Div([
                        html.Label("レアカテゴリをまとめる:", style={"color": "#34495e", "marginLeft": "10px"}),
                        dbc.Switch(
                            id="rare-category-switch",
                            value=False,  # 初期状態では無効
                            style={"marginLeft": "10px"}
                        )
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginTop": "20px"}),
                    html.Div(dropdown_rows)
                ]
            )
        else:
            cat_summary_dropdown = html.Div(
                children=[
                    html.Div([
                        dbc.Switch(id="rare-category-switch", value=False)
                    ], style={"display": "none"}),
                ]
            )
    else:
        cat_summary_table = ""       # カテゴリデータがないときは表示しない    
        cat_summary_dropdown = html.Div(
            children=[
                html.Div([
                    dbc.Switch(id="rare-category-switch", value=False)
                ], style={"display": "none"}),
            ]
        )

    return num_summary_table, cat_summary_table, cat_summary_dropdown

@app.callback(
    [Output({"type": "dynamic-dropdown", "col": MATCH}, "options"),
     Output({"type": "dynamic-dropdown", "col": MATCH}, "value")],
    [Input("rare-category-switch", "value")],
    [State({"type": "dynamic-dropdown", "col": MATCH}, "id"),
     State({"type": "dynamic-dropdown", "col": MATCH}, "value"),
     State("store-original-data", "data")],
    prevent_initial_call=True
)
def update_dropdown_options(rare_category_switch, dropdown_id, current_value, stored_data):
    if stored_data is None:
        return no_update, no_update

    col = dropdown_id["col"]
    df = pd.DataFrame(stored_data["data"])
    if col not in df.columns:
        return no_update, no_update

    categories = df[col].dropna().unique()
    n_total = len(df)
    rare_threshold = int(np.ceil(np.sqrt(n_total) / 5))
    value_counts = df[col].value_counts(dropna=True)
    rare_cats = value_counts[value_counts <= rare_threshold].index.tolist()

    merge_active = rare_category_switch and len(rare_cats) > 1 and len(rare_cats) < len(value_counts)

    if merge_active:
        display_cats = [c for c in categories if c not in rare_cats]
        display_cats.append("その他")
        display_cats = sorted(list(set(display_cats)))
    else:
        display_cats = sorted(list(categories))

    options = [
        {"label": "しない", "value": "しない"},
        {"label": "目的変数", "value": "目的変数"}
    ]
    options += [{"label": f"説明変数(基準:{cat})", "value": f"説明変数(基準:{cat})"} for cat in display_cats]

    new_value = current_value
    valid_values = [opt["value"] for opt in options]
    if current_value not in valid_values:
        if current_value.startswith("説明変数(基準:") and current_value[8:-1] in rare_cats:
            new_value = "説明変数(基準:その他)"
        else:
            new_value = "しない"

    return options, new_value

# エンコード設定
@app.callback(
    [Output("store-num-data", "data"),
     Output("store-class-data", "data"),
     Output("store-label-data", "data"),
     Output("impute-option", "style"),
     Output("store-original-categories", "data"),
     Output("store-rare-categories", "data")],
    [Input("store-original-data", "data"),
     Input("rare-category-switch", "value"),
     Input({"type": "dynamic-dropdown", "col": ALL}, "value"),
     Input("impute-mode", "value")],
    State({"type": "dynamic-dropdown", "col": ALL}, "id")
)
def encode_data(stored_data, rare_category_switch, dropdown_values, impute_method, dropdown_ids):
    if stored_data is None:
        return None, None, None, {"display": "none"}, None, None

    df = pd.DataFrame(stored_data["data"])
    df.columns = stored_data["columns"]

    # カテゴリ変数を抽出
    df_category = df.select_dtypes(exclude=[np.number])
    cat_summary_df = pd.DataFrame({"変数": df_category.columns})
    label_data = df_category.copy()
    if "行番号" in df.columns:
        num_data = df.select_dtypes(include=[np.number]).drop(columns=["行番号"], errors="ignore").copy()  # num_data から除外
        label_data.insert(0, "行番号", df["行番号"])  # label_data の先頭に追加
    else:
        num_data = df.select_dtypes(include=[np.number]).copy()
    class_columns = df_category.columns[df_category.apply(lambda col: col.dropna().isin(["O","X"]).all(), axis=0)]
    class_data = df_category[class_columns].copy()

    # レアカテゴリ統合のために df をコピーして加工
    df_processed = df.copy()

    # カテゴリ変数の欠損補完:
    # 欠損補完はCVリークを防ぐため各CV fold内のPipelineで実行する。
    # ただし UI 表示・ダミー列用には "missing" / "iterative_missing" のとき "欠損" ラベルを適用可能。
    for col in df_category.columns:
        if impute_method in ["iterative_missing", "missing"]:
            df_processed[col] = df_processed[col].fillna("欠損")

    original_categories = {col: sorted(df[col].dropna().unique().tolist()) for col in df_category.columns}

    option_map = { id_obj["col"]: value for id_obj, value in zip(dropdown_ids, dropdown_values) }

    n_total = len(df)
    rare_threshold = int(np.ceil(np.sqrt(n_total) / 5))
    rare_category_map = {}
    base_categories_map = {}
    
    for col in cat_summary_df["変数"]:
        if col == "__dummy__": 
            continue  # ダミー列はスキップ
        
        option = option_map.get(col, "しない")
        rare_category_map[col] = []
     
        # レアカテゴリ統合
        if (option == "目的変数" or option.startswith("説明変数(基準:")) and rare_category_switch:
            value_counts = df[col].value_counts(dropna=True)
            rare_cats = value_counts[value_counts <= rare_threshold].index.tolist()

            # レアカテゴリが1つ以下 or 全カテゴリがレア → 統合スキップ
            if len(rare_cats) <= 1 or len(rare_cats) == len(value_counts):
                pass # 何もしない
            else:
                df_processed[col] = df_processed[col].mask(df_processed[col].isin(rare_cats), "その他")
                rare_category_map[col] = rare_cats
                
        # One-hot エンコード
        if option == "しない":
            continue  # 元の label_data に残すだけ
        elif option == "目的変数":
            # dropなし One-hot → 0/1 を X/O に変換
            dummies = pd.get_dummies(df_processed[col], prefix="", prefix_sep="", drop_first=False)
            dummies.columns = [f"{col}[{c}]" for c in dummies.columns]
            dummies.loc[df[col].isna(), :] = np.nan
            dummies = dummies.replace({0: "X", 1: "O"})
            class_data = pd.concat([class_data, dummies], axis=1)
        elif option.startswith("説明変数(基準:"):
            # "説明変数(基準:○○)" の ○○ 部分だけ抜き出す
            base = option[8:-1]
            base_categories_map[col] = base
            # drop-firstはアルファベット順で最初のカテゴリを落とすので、自作 drop をする
            dummies = pd.get_dummies(df_processed[col], prefix="", prefix_sep="", drop_first=False).astype(float)
            dummies.columns = [f"{col}[{c}]" for c in dummies.columns]
            # 「線形モデル補完/欠損埋め」では、欠損を事前に「欠損」カテゴリへ置換済み。
            # その行を全ダミーNaNに戻すと [欠損] が1にならないため、マスクしない。
            if impute_method not in ["iterative_missing", "missing"]:
                dummies.loc[df[col].isna(), :] = np.nan
            # 基準カテゴリ列だけ削除
            base_col = f"{col}[{base}]"
            if base_col in dummies.columns:
                dummies = dummies.drop(columns=[base_col])
            num_data = pd.concat([num_data, dummies], axis=1)
        else:
            # 不明な選択肢はスキップ
            continue

    # 数値データに欠損があるかどうかを判定
    if num_data.isnull().any().any():
        impute_style = {"display": "flex", "alignItems": "center", "marginTop": "10px"}
    else:
        impute_style = {"display": "none"}
    
    # 既存の rare_category_map と base_categories を辞書にまとめて保存
    rare_category_store_data = {
        "rare_map": rare_category_map,
        "base_cats": base_categories_map,
        "enabled": bool(rare_category_switch)
    }
    # 後方互換のため、辞書直下にも各列の rare_cats を保持させる
    for k, v in rare_category_map.items():
        rare_category_store_data[k] = v

    return (
        {"data": num_data.to_dict("records"), "columns": list(num_data.columns)},
        {"data": class_data.to_dict("records"), "columns": list(class_data.columns)},
        {"data": label_data.to_dict("records"), "columns": list(label_data.columns)},
        impute_style,
        {"data": original_categories},
        {"data": rare_category_store_data},
    )

@app.callback(
    [Output("corr-matrix", "figure"),
     Output("eta-cramers-matrix", "figure"),
     Output("vif-matrix", "figure"),
     Output("corr-top", "style"),
     Output("corr-bottom", "style"),
     Output("vif-accordion", "style")],
    [Input("store-original-data", "data")]
)
def upload_analysis(stored_data):
    if stored_data is None:
        return (
            default_fig("相関係数行列", ""),
            default_fig("相関比 / クラメールのV", ""),
            default_fig("多重共線性ヒートマップ", ""),
            {"display": "none"},
            {"display": "none"},
            {"display": "none"}
        )
    
    # DataFrameに復元
    df = pd.DataFrame(stored_data["data"])
    df.columns = stored_data["columns"]
    
    num_data = df.select_dtypes(include=[np.number]).drop(columns=["行番号"], errors="ignore").copy()  # 数値データのみを選択

    # 数値以外を抽出し、ユニーク率70%未満の列だけ残す
    cat_candidates = df.select_dtypes(exclude=[np.number])
    cat_data = pd.DataFrame()
    for col in cat_candidates.columns:
        n_data = df[col].count()
        if n_data > 0:
            unique_ratio = df[col].nunique() / n_data
            if unique_ratio < 0.7:
                cat_data[col] = df[col]

    if num_data.shape[1] > 0:
        # 相関行列の作成
        correlation_matrix = num_data.corr()
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale=[[0, "blue"], [0.5, "white"], [1, "red"]],
            zmin=-1,
            zmax=1,
            title="相関係数行列"
        )
        fig_corr.update_layout(
            #dragmode="pan",
            coloraxis=dict(colorbar_title="相関係数"),
            plot_bgcolor="darkgray",
            xaxis=dict(constrain="domain", showgrid=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(constrain="domain", showgrid=False)
        )
        fig_corr.update_traces(hovertemplate="<b>x: %{x}</b><br><b>y: %{y}</b><br><b>相関係数: %{z}</b><extra></extra>")
        corr_top = {"display": "block"}

        # VIF行列を計算する関数
        def calculate_vif(data):
            data_filled = data.fillna(data.mean())  # 平均値で補完
            features = data_filled.columns
            Xc = data_filled.values
            n = Xc.shape[1]
            model = Ridge(alpha=0.1)
            
            # データの標準化
            scaler = RobustScaler()
            Xc_standardized = scaler.fit_transform(Xc)
        
            rc = np.empty((n, n))
            vif_values = []
            r2_values = []
            X = np.copy(Xc_standardized)
            for i in range(n):
                y = np.copy(X[:, i])  # ターゲット変数
                X[:, i] = 1
        
                model.fit(X, y)
                y_pred = model.predict(X)
        
                score = r2_score(y, y_pred)    # R²スコアを計算
                if score < 1:
                    vif_values.append(1 / (1 - score))  # VIF値を保存
                else:
                    vif_values.append(np.nan)
                r2_values.append(score)  # R²値を保存
                rc[i] = model.coef_ #/ (np.linalg.norm(model.coef_) + 1e-15) * score
                X[:, i] = y  # 元に戻す
        
            # 回帰係数行列をpd.DataFrame形式に変換
            vif_matrix = pd.DataFrame(rc, index=features, columns=features).iloc[::-1, :]
            vif_table = pd.DataFrame({"説明変数": features, "VIF値": vif_values, "R²値": r2_values})
            vif_table = vif_table.sort_values(by="VIF値", ascending=False)
            return vif_matrix, vif_table
        
        # VIFを計算
        vif_matrix, vif_table = calculate_vif(num_data)
        
        # プロットを作成
        fig_vif = sp.make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "xy"}, {"type": "table"}]],
            subplot_titles=("多重共線性ヒートマップ", "VIF値(降順)")
        )
        
        # VIFヒートマップを追加
        heatmap = go.Heatmap(
            z=vif_matrix.values,
            x=vif_matrix.columns,
            y=vif_matrix.index,
            colorscale=[[0, "blue"], [0.5, "white"], [1, "red"]],
            zmin=-1,
            zmax=1,
            showscale=False,
            text=np.vectorize(lambda v: f"{v:.2f}")(vif_matrix.values),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        )
        
        # VIF値の表を追加
        table = go.Table(
            columnwidth=[2, 1, 1],
            header=dict(values=["説明変数", "VIF値", "R²値"], fill_color="lightgray", align="center"),
            cells=dict(values=[
                vif_table["説明変数"], 
                vif_table["VIF値"].round(2),
                vif_table["R²値"].round(3)
            ],align=["left", "right", "right"])
        )
    
        fig_vif.add_trace(heatmap, row=1, col=1)
        fig_vif.add_trace(table, row=1, col=2)
        #fig_vif.update_layout(dragmode="pan")
        vif_acc = {"display": "block"}
    else:
        fig_corr = default_fig("相関係数行列", "")
        fig_vif = default_fig("多重共線性ヒートマップ", "")
        corr_top = {"display": "none"}
        vif_acc = {"display": "none"}

    if cat_data.shape[1] > 0:
        # 相関比（カテゴリ vs 数値）
        def correlation_ratio(categories, values):
            # カテゴリと数値を横に結合して、両方欠損のない行だけ残す
            aligned = pd.concat([pd.Series(categories), pd.Series(values)], axis=1).dropna()
            categories = aligned.iloc[:, 0]
            values = aligned.iloc[:, 1]
    
            if len(values) == 0:
                return np.nan
    
            fcat, _ = pd.factorize(categories)
            cat_num = np.max(fcat) + 1
            y_avg = np.mean(values)
    
            numerator = 0
            denominator = np.sum((values - y_avg) ** 2)
    
            for i in range(cat_num):
                cat_values = values.iloc[np.argwhere(fcat == i).flatten()]
                if len(cat_values) > 0:
                    numerator += len(cat_values) * (np.mean(cat_values) - y_avg) ** 2
    
            return numerator / denominator if denominator != 0 else 0
    
        # クラメールのVを計算する関数
        def cramers_v(x, y):
            # 欠損は除外（カテゴリ化しない）
            aligned = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
            x = aligned.iloc[:, 0]
            y = aligned.iloc[:, 1]
    
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            r, k = confusion_matrix.shape
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))  #補正あり
            #return np.sqrt(phi2 / min(k-1, r-1))  #補正なし
    
        # 軸の定義（x:カテゴリ, y:数値＋カテゴリ）
        all_x = list(cat_data.columns)                           # カテゴリ
        all_y = list(num_data.columns) + list(cat_data.columns)  # 数値＋カテゴリ
        
        matrix = pd.DataFrame(index=all_y, columns=all_x)
    
        for x in all_x:  # 列:カテゴリ変数
            for y in all_y:  # 行:数値 or カテゴリ
                if y in num_data.columns:  # 数値 vs カテゴリ → 相関比
                    matrix.loc[y, x] = correlation_ratio(cat_data[x], num_data[y])
                else:  # カテゴリ vs カテゴリ → クラメールのV
                    matrix.loc[y, x] = cramers_v(cat_data[x], cat_data[y])
    
        matrix = matrix.astype(float)
    
        fig_cat = px.imshow(
            matrix,
            text_auto=".2f",
            color_continuous_scale=[[0, "white"], [1, "red"]],
            zmin=0,
            zmax=1,
            title="相関比 / クラメールのV"
        )
        fig_cat.update_layout(
            coloraxis=dict(colorbar_title="関連度"),
            plot_bgcolor="darkgray",
            xaxis=dict(side="bottom")
        )
        fig_cat.update_traces(
            hovertemplate="<b>x: %{x}</b><br><b>y: %{y}</b><br><b>z: %{z}</b><extra></extra>"
        )
        corr_bottom = {"display": "block"}
    else:
        fig_cat = default_fig("相関比 / クラメールのV", "")
        corr_bottom = {"display": "none"}

    return fig_corr, fig_cat, fig_vif, corr_top, corr_bottom, vif_acc

# コールバック: 相関係数行列クリックで散布図表示
@app.callback(
    Output('pair-plot', 'figure'),
    [Input('corr-matrix', 'clickData'),
    Input("store-num-data", "data"),
    Input("store-label-data", "data"),
    Input("color-toggle", "value")],
    State("label-variables", "value")
)
def update_plot(clickData, stored_data, label_data, color_toggle, label_variable):
    if clickData is None:
        return default_fig("データ分布","相関係数行列のセルをクリックで表示")

    data = pd.DataFrame(stored_data["data"])
    label_data = pd.DataFrame(label_data["data"])
    label_data = label_data.fillna("データなし")
    specific_labels = ["X", "データなし", "O"]  # 分類のラベル順序を定義
    # ラベルの順序と数値割り当てを作成
    if set(label_data[label_variable].unique()).issubset(specific_labels):  # 分類の場合
        label_mapping = {label: idx for idx, label in enumerate(specific_labels)}
    else:
        unique_labels = label_data[label_variable].unique()
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = label_data[label_variable].map(label_mapping)  # 数値ラベルに変換
    
    point = clickData['points'][0]
    x_data = data[point['x']]  # クリックされたセルの列
    y_data = data[point['y']]  # クリックされたセルの行
    
    # プロットの色分け用設定
    if color_toggle and label_variable:
        color = numeric_labels  # 数値化されたラベルで色分け
    else:
        color = "blue"  # スイッチが無効の場合はデフォルト色

    hover_label = label_variable if label_variable else "ラベル"
    hovertemplate = (
        f"<b>{hover_label}:</b> "  # ラベル名を含める
        "%{text}<br><b>x:</b> %{x}<br><b>y:</b> %{y}<extra></extra>"
    )

    # 散布図
    scatter = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        name="散布図",
        marker=dict(color=color, colorscale="Bluered"),
        text=label_data.loc[data.index, label_variable] if label_variable else data.index + "行目",  # ラベル変数がない場合は行番号を表示
        hovertemplate=hovertemplate
    )
    # x軸のヒストグラム
    hist_x = go.Histogram(
        x=x_data,
        name="X軸ヒストグラム",
        marker=dict(color="lightblue"),
        yaxis="y2"  # 別のy軸を使用して配置
    )
    # y軸のヒストグラム
    hist_y = go.Histogram(
        y=y_data,
        name="Y軸ヒストグラム",
        marker=dict(color="lightpink"),
        xaxis="x2"  # 別のx軸を使用して配置
    )
    # 図の作成
    layout = go.Layout(
        title=f"データ分布: {point['y']} vs {point['x']}",
        xaxis=dict(title=point['x'],domain=[0, 0.8]),
        yaxis=dict(title=point['y'],domain=[0, 0.8]),
        xaxis2=dict(domain=[0.82, 1]),  # x軸上にヒストグラムを配置
        yaxis2=dict(domain=[0.82, 1]),  # y軸にヒストグラムを配置
        showlegend=False,
    )
    fig_pairplot = go.Figure(data=[scatter, hist_x, hist_y], layout=layout)
    fig_pairplot.update_layout(
        #dragmode="pan"
    )

    return fig_pairplot

# コールバック: 相関比/クラメールのVのセルクリックで図表示
@app.callback(
    Output('crosstab', 'figure'),
    [Input('eta-cramers-matrix', 'clickData'),
    Input("store-num-data", "data"),
    Input("store-label-data", "data"),
    Input("color-toggle", "value")],
    State("label-variables", "value")
)
def update_crosstab(clickData, num_data, cat_data, color_toggle, label_variable):
    if clickData is None:
        return default_fig("関連度プロット", "相関比/クラメールのVのセルをクリックで表示")

    num_df = pd.DataFrame(num_data["data"])
    raw_cat_df = pd.DataFrame(cat_data["data"])
    # ユニーク率を計算して 70% 未満の列だけ残す
    cat_df = pd.DataFrame()
    for col in raw_cat_df.columns:
        unique_ratio = raw_cat_df[col].nunique() / len(raw_cat_df)
        if unique_ratio < 0.7:   # ユニーク率が 70% 未満ならカテゴリとして採用
            cat_df[col] = raw_cat_df[col]

    point = clickData['points'][0]
    x_name = point['x']
    y_name = point['y']

    # ケース1: xがカテゴリ, yが数値 → 箱ひげ図
    if x_name in cat_df.columns and y_name in num_df.columns:
        x_series = cat_df[x_name].astype(object).fillna("欠損")  # 欠損をカテゴリ化
        y_series = num_df[y_name]                                # 数値

        # 数値側の欠損は描画できないので除外
        mask = ~y_series.isna()
        x_plot = x_series[mask]
        y_plot = y_series[mask]

        # カテゴリ順で欠損を最後に並べる
        uniques = pd.Index(x_plot.unique())
        ordered = [c for c in uniques if c != "欠損"] + (["欠損"] if "欠損" in uniques else [])

        fig = go.Figure()
        fig.add_trace(go.Box(
            x=x_plot,
            y=y_plot,
            boxpoints="all",
            jitter=0.3,
            marker_color="blue",
        ))
        fig.update_layout(
            title=f"箱ひげ図: {y_name} vs {x_name}",
            xaxis_title=x_name,
            yaxis_title=y_name,
            xaxis=dict(categoryorder="array", categoryarray=ordered)
        )
        return fig

    # ケース2: xもyもカテゴリ → 100%積み上げ棒グラフ
    elif x_name in cat_df.columns and y_name in cat_df.columns:
        x_series = cat_df[x_name].fillna("欠損")
        y_series = cat_df[y_name].fillna("欠損")
    
        ct = pd.crosstab(y_series, x_series, normalize="index") * 100
    
        # 欠損を最後に並べ替え
        if "欠損" in ct.columns:
            cols = [c for c in ct.columns if c != "欠損"] + ["欠損"]
            ct = ct[cols]
        if "欠損" in ct.index:
            rows = [r for r in ct.index if r != "欠損"] + ["欠損"]
            ct = ct.loc[rows]
    
        fig = px.bar(
            ct,
            barmode="stack",
            title=f"構成比: {y_name} ごとの {x_name} (%)"
        )
        fig.update_layout(
            xaxis_title=y_name,
            yaxis_title="割合 (%)",
            legend_title=x_name
        )
        return fig

    # その他（想定外）
    else:
        return default_fig("関連度プロット", "この組み合わせには対応していません")

# エンコード変数のグループ化用関数
def group_encoded_columns(columns):
    groups = {}
    for col in columns:
        if "[" in col and col.endswith("]"):
            base = col.split("[")[0]
            groups.setdefault(base, []).append(col)
        else:
            groups.setdefault(col, [col])
    return groups

# コールバック: チェックボックスオプションを生成し、選択を制御
@app.callback(
    [Output("option-explanatory", "options"),
     Output("option-target", "options"),
     Output("option-class", "options"),
     Output("label-variables", "options")],
    [Input("store-num-data", "data"),
     Input("store-class-data", "data"),
     Input("store-label-data", "data"),
     Input("option-explanatory", "value"),
     Input("option-target", "value"),
     Input("impute-mode", "value")]
)
def update_display_options(stored_data, stored_class_data, stored_label_data,
                           explanatory_values, target_values, impute_method):
    if stored_data is None:
        dummy = [{"label": "データなし", "value": "none"}]
        return dummy, dummy, dummy, dummy

    # 数値データ（エンコード後）
    num_data = pd.DataFrame(stored_data["data"])
    groups = group_encoded_columns(num_data.columns)

    # 欠損判定（元変数単位で判定）
    columns = []
    for base, cols in groups.items():
        has_na = num_data[cols].isnull().any().any()
        # impute-mode が none のときだけ欠損列を disabled にする
        disabled_flag = (impute_method == "none") and has_na
        columns.append({"label": base, "value": base, "disabled": disabled_flag})

    # 相互排他設定
    explanatory_options = [
        {"label": col["label"], "value": col["value"],
         "disabled": col["disabled"] or col["value"] in target_values}
        for col in columns
    ]
    target_options = [  # 目的変数（列名に "[" を含まない）
        {"label": col, "value": col, "disabled": col in explanatory_values}
        for col in num_data.columns if "[" not in col
    ]

    # クラスデータ
    class_data = pd.DataFrame(stored_class_data["data"])
    class_groups = group_encoded_columns(class_data.columns)
    option_class = [{"label": k, "value": k} for k in class_groups.keys()]

    # ラベルデータ
    label_data = pd.DataFrame(stored_label_data["data"])
    columns = [{"label": col, "value": col} for col in label_data.columns]
    option_label = [{"label": col["label"], "value": col["value"]} for col in columns]

    return explanatory_options, target_options, option_class, option_label

@app.callback(
    Output("label-variables", "value"),
    Input("store-label-data", "data")
)
def set_initial_label(stored_label_data):
    if stored_label_data is None:
        return None

    label_data = pd.DataFrame(stored_label_data["data"])
    initial_label_value = "行番号" if "行番号" in label_data.columns else None
    return initial_label_value

#選択肢→解析する変数に展開
@app.callback(
    [Output("explanatory-variables", "value"),
     Output("target-variables", "value"),
     Output("class-variables", "value")],
    [Input("option-explanatory", "value"),
     Input("option-target", "value"),
     Input("option-class", "value")],
    [State("store-num-data", "data"),
     State("store-class-data", "data"),
     State("store-label-data", "data")]
)
def expand_selected(display_explanatory, display_target, display_class,
                    stored_data, stored_class_data, stored_label_data):
    def expand(values, groups):
        expanded = []
        if values is None:
            return []
        for v in values:
            expanded += groups.get(v, [v])
        return expanded

    if not stored_data or "data" not in stored_data:
            return [], [], []
    if not stored_class_data or "data" not in stored_class_data:
        return [], [], []
    num_data = pd.DataFrame(stored_data["data"])
    groups_num = group_encoded_columns(num_data.columns)

    class_data = pd.DataFrame(stored_class_data["data"])
    groups_class = group_encoded_columns(class_data.columns)

    return (
        expand(display_explanatory, groups_num),
        expand(display_target, groups_num),
        expand(display_class, groups_class)
    )

# コールバック: 全選択ボタン
@app.callback(
    [Output("option-explanatory", "value"),
     Output("option-target", "value"),
     Output("option-class", "value")],
    [Input("select-all-explanatory", "n_clicks"),
     Input("select-all-target", "n_clicks"),
     Input("select-all-class", "n_clicks"),
     Input("store-num-data", "data"),
     Input("impute-mode", "value")],
    [State("option-explanatory", "options"),
     State("option-target", "options"),
     State("option-class", "options"),
     State("option-explanatory", "value"),
     State("option-target", "value"),
     State("option-class", "value")]
)
def handle_variables(n_clicks_explanatory, n_clicks_target, n_clicks_class, stored_data, impute_method,
                     explanatory_options, target_options, class_options, explanatory_values, target_values, class_values):
    # どのボタンが押されたかを判定
    triggered_id = ctx.triggered_id

    if stored_data is None:
        return [], [], []
    if triggered_id == "store-num-data":  # 選択肢が変わったら選択リセット
        return [], [], []
    
    # 欠損のある変数だけチェックを外す処理
    if impute_method == "none":
        num_data = pd.DataFrame(stored_data["data"])
        groups = group_encoded_columns(num_data.columns)

        # 欠損のある変数を除外
        explanatory_values = [
            v for v in explanatory_values
            if not num_data[groups[v]].isnull().any().any()
        ]
    def filter_valid_values(options, current_values):
        # オプションから選択可能な値のみ抽出し、無効な選択を削除
        valid_values = [opt["value"] for opt in options if not opt.get("disabled", False)]
        return [value for value in current_values if value in valid_values]

    # 現在の選択値をフィルタリングして有効な値のみ残す
    explanatory_values = filter_valid_values(explanatory_options, explanatory_values)
    target_values = filter_valid_values(target_options, target_values)

    # 全選択ボタンの処理
    if triggered_id == "select-all-explanatory":
        if len(explanatory_values) == len([opt for opt in explanatory_options if not opt.get("disabled", False)]):
            return [], target_values, class_values  # 全て選択されている場合は解除
        else:
            return [opt["value"] for opt in explanatory_options if not opt.get("disabled", False)], target_values, class_values
    elif triggered_id == "select-all-target":
        if len(target_values) == len([opt for opt in target_options if not opt.get("disabled", False)]):
            return explanatory_values, [], class_values  # 全て選択されている場合は解除
        else:
            return explanatory_values, [opt["value"] for opt in target_options if not opt.get("disabled", False)], class_values
    elif triggered_id == "select-all-class":
        if len(class_values) == len(class_options):
            return explanatory_values, target_values, []  # 全て選択されている場合は解除
        else:
            return explanatory_values, target_values, [opt["value"] for opt in class_options]

    return explanatory_values, target_values, class_values

# コールバック: CV解析
@app.callback(
    [Output({"type": "store-r2-scores", "index": ALL}, "data"),
     Output({"type": "grid-search-score", "index": ALL}, "figure")],
    Input("analyze-button", "n_clicks"),
    [State("store-num-data", "data"),
     State("store-class-data", "data"),
     State("explanatory-variables", "value"),
     State("target-variables", "value"),
     State("class-variables", "value"),
     State("regression-method", "value"),
     State("impute-mode", "value"),
     State("store-original-categories", "data"),
     State("store-rare-categories", "data"),
     State("store-original-data", "data")]
)
def perform_analysis(n_clicks, stored_data, stored_class_data, explanatory_variables, target_variables, class_variables, 
                     regression_method, impute_method, original_categories, rare_category_store, stored_original_data):

    total_variables = target_variables + class_variables
    
    if n_clicks == 0 or stored_data is None or not explanatory_variables or (not target_variables and not class_variables):
        return [{"target": target, "r2_scores": []} for target in total_variables], [default_fig("グリッドサーチ結果", "")] * len(total_variables)
    else:
        data = pd.DataFrame(stored_data["data"])
        raw_df = pd.DataFrame(stored_original_data["data"], columns=stored_original_data["columns"]) if stored_original_data else data.copy()
        cv_X_all = build_cv_feature_frame(raw_df, data, explanatory_variables)
        selected_data = data[explanatory_variables + target_variables]
        class_data = pd.concat([data[explanatory_variables], pd.DataFrame(stored_class_data["data"])], axis=1)

        rare_category_map = {}
        base_categories_map = {}
        if rare_category_store and "data" in rare_category_store:
            rdata = rare_category_store["data"]
            rare_category_map = rdata.get("rare_map", rdata)
            base_categories_map = rdata.get("base_cats", {})
        all_categories_map = {}  # fold内の学習データからカテゴリを学習する
        rare_category_enabled = bool(rare_category_store and rare_category_store.get("data", {}).get("enabled", False))

        if regression_method in ["quantile", "huber"]:
            return [
                        [{"target": target, "r2_scores": []} for target in total_variables],
                        [default_fig("グリッドサーチ結果", "")] * len(total_variables)
            ]

        else:
            # 回帰手法に応じた alpha_values と l1_ratios の設定
            if regression_method == "elasticnet":
                alpha_values = np.logspace(-5, 3, num=9)
                l1_ratios = np.linspace(0, 1.0, num=11)
            elif regression_method == "lasso":
                alpha_values = np.logspace(-5, 3, num=33)  # LASSO用の値
                l1_ratios = [1.0]  # LASSOは l1_ratio が固定
            elif regression_method == "ridge":
                alpha_values = np.logspace(-5, 3, num=33)  # Ridge用の値
                l1_ratios = [0.0]  # Ridgeは l1_ratio が固定
    
            store_r2_scores = []
            grid_search_figures = []
    
            # 回帰
            for target in target_variables:
                dropped_data = selected_data.dropna(subset=[target])
                X = cv_X_all.loc[dropped_data.index].copy()
                y = dropped_data[target].values
                r2_scores = np.zeros((len(alpha_values), len(l1_ratios)))
    
                n_fold = int(np.floor(min(len(y), 10, 5 + 100 / np.sqrt(len(y)))))
                kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)

                # ---- Foldごとに補完＆スケーリングをキャッシュ ----
                fold_data = []
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
            
                    # 前処理
                    preprocessor = NumericPreprocessor(
                        impute_method=impute_method,
                        scale=True,
                        base_categories=base_categories_map,
                        rare_category_map=rare_category_map,
                        all_categories=None,
                        rare_category_enabled=rare_category_enabled
                    )
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
            
                    # キャッシュに保存
                    fold_data.append((X_train_processed, X_test_processed, y_train, y_test))
            
                # ---- キャッシュ済みデータを使ってパラメータ探索 ----
                for i, alpha in enumerate(alpha_values):
                    for j, l1_ratio in enumerate(l1_ratios):
                        all_y_test, all_y_pred = [], []
            
                        for X_train_processed, X_test_processed, y_train, y_test in fold_data:
                            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
                            model.fit(X_train_processed, y_train)
                            y_pred = model.predict(X_test_processed)
            
                            all_y_test.extend(y_test)
                            all_y_pred.extend(y_pred)
            
                        r2_scores[i, j] = concordance_corrcoef(all_y_test, all_y_pred)
                store_r2_scores.append({"target": target, "r2_scores": r2_scores.tolist()})
    
                # CVヒートマップ
                fig_cv = px.imshow(
                    r2_scores,
                    x=[f"{l1_ratio:.1f}" for l1_ratio in l1_ratios],
                    y=[f"{alpha:.5f}" for alpha in alpha_values],
                    labels={"x": "l1_ratio", "y": "alpha", "color": "適合度"},
                    text_auto=".3f",
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
                
                # x軸ラベルの表示制御
                if regression_method != "elasticnet":
                    fig_cv.update_layout(
                        xaxis_title="",  # ラベルを空にする
                        xaxis=dict(showticklabels=False)  # 目盛ラベルも非表示にする
                    )
                
                fig_cv.update_layout(
                    title=f"{target}のグリッドサーチ結果 (検証データへのモデル適合度)",
                    #dragmode="pan"
                )
                grid_search_figures.append(fig_cv)
            
            # 分類
            for c_target in class_variables:
                dropped_data = class_data.dropna(subset=[c_target])
                dropped_data = dropped_data.copy()
                dropped_data.loc[:, c_target] = dropped_data[c_target].map({"O": 1, "X": 0})
                X = cv_X_all.loc[dropped_data.index].copy()
                y = dropped_data[c_target].astype(int).values
                r2_scores = np.zeros((len(alpha_values), len(l1_ratios)))
    
                min_class_count = np.min(np.bincount(y))
                n_fold = int(np.floor(min(min_class_count, 10, 5 + 100 / np.sqrt(len(y)))))
                kf = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=1, random_state=0)

                # ---- Foldごとに補完＆スケーリングをキャッシュ ----
                fold_data = []
                for train_idx, test_idx in kf.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                
                    # 欠損補完
                    preprocessor = NumericPreprocessor(
                        impute_method=impute_method,
                        scale=True,
                        base_categories=base_categories_map,
                        rare_category_map=rare_category_map,
                        all_categories=None,
                        rare_category_enabled=rare_category_enabled
                    )
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                
                    # キャッシュ保存
                    fold_data.append((X_train_processed, X_test_processed, y_train, y_test))
                
                # ---- キャッシュ済みデータを使ってパラメータ探索 ----
                for i, alpha in enumerate(alpha_values):
                    for j, l1_ratio in enumerate(l1_ratios):
                        all_y_test, all_y_pred = [], []
                
                        for X_train_processed, X_test_processed, y_train, y_test in fold_data:
                            model = LogisticRegression(
                                penalty="elasticnet",
                                solver="saga",
                                C=1/alpha,
                                l1_ratio=l1_ratio,
                                class_weight='balanced',
                                random_state=0
                            )
                            model.fit(X_train_processed, y_train)
                            y_pred = model.predict_proba(X_test_processed)[:, 1]
                
                            all_y_test.extend(y_test)
                            all_y_pred.extend(y_pred)
                
                        r2_scores[i, j] = roc_auc_score(all_y_test, all_y_pred)
                store_r2_scores.append({"target": c_target, "r2_scores": r2_scores.tolist()})
    
                # CVヒートマップ
                fig_cv = px.imshow(
                    r2_scores,
                    x=[f"{l1_ratio:.1f}" for l1_ratio in l1_ratios],
                    y=[f"{alpha:.5f}" for alpha in alpha_values],
                    labels={"x": "l1_ratio", "y": "alpha", "color": "ROC-AUC"},
                    text_auto=".3f",
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
                
                # x軸ラベルの表示制御
                if regression_method != "elasticnet":
                    fig_cv.update_layout(
                        xaxis_title="",  # ラベルを空にする
                        xaxis=dict(showticklabels=False)  # 目盛ラベルも非表示にする
                    )
                
                fig_cv.update_layout(
                    title=f"{c_target}のグリッドサーチ結果 ({n_fold}-fold CVのROC-AUC)",
                    #dragmode="pan"
                )
                grid_search_figures.append(fig_cv)
    
            return store_r2_scores, grid_search_figures

# コールバック: スライダーの値を更新
@app.callback(
    [Output({"type": "alpha-slider", "index": MATCH}, "value"),
     Output({"type": "l1-ratio-slider", "index": MATCH}, "value")],
    [Input({"type": "grid-search-score", "index": MATCH}, "clickData"),
     Input({"type": "store-r2-scores", "index": MATCH}, "data")],
    [State("regression-method", "value")]
)
def update_sliders(click_data, r2_scores, regression_method):

    if r2_scores is None or regression_method in ["quantile", "huber"]:
        return no_update, 0.0

    try:
        r2_scores_array = np.array(r2_scores["r2_scores"])  # リスト形式を配列へ変換

        if click_data:
            # クリックされたセルの位置を取得
            clicked_point = click_data["points"][0]
            selected_alpha = np.log10(float(clicked_point["y"]))
            selected_l1_ratio = float(clicked_point["x"])
        else:
            # R²スコアの中で最大値を検索
            if regression_method == "elasticnet":
                max_idx = np.unravel_index(np.argmax(r2_scores_array), r2_scores_array.shape)
                selected_alpha = np.log10(np.logspace(-5, 3, num=9)[max_idx[0]])  # alphaの値を取得
                selected_l1_ratio = np.linspace(0, 1.0, num=11)[max_idx[1]]  # l1_ratioの値を取得
            elif regression_method == "lasso":
                max_idx = np.argmax(r2_scores_array)  # LASSOでは1次元
                selected_alpha = np.log10(np.logspace(-5, 3, num=33)[max_idx])
                selected_l1_ratio = 1.0  # 固定値
            elif regression_method == "ridge":
                max_idx = np.argmax(r2_scores_array)  # Ridgeでは1次元
                selected_alpha = np.log10(np.logspace(-5, 3, num=33)[max_idx])
                selected_l1_ratio = 0.0  # 固定値
            else:
                selected_l1_ratio = 0.0  # l1_ratio不要

        return selected_alpha, selected_l1_ratio
    except Exception:
        return no_update, no_update

# コールバック: リアルタイム解析
@app.callback(
    [Output({"type": "saved-cv-scores", "index": ALL}, "data"),
     Output({"type": "result-message", "index": ALL}, "children"),
     Output({"type": "scatter-plot", "index": ALL}, "figure"),
     Output({"type": "feature-importance-plot", "index": ALL}, "figure"),
     Output({"type": "cross-validation-plot", "index": ALL}, "figure"),
     Output({"type": "cv-export-table", "index": ALL}, "data"),
     Output({"type": "test-table", "index": ALL}, "data")],
    [Input("dynamic-tabs", "children"),
     Input({"type": "alpha-slider", "index": ALL}, "value"),
     Input({"type": "l1-ratio-slider", "index": ALL}, "value"),
     Input({"type": "threshold-slider", "index": ALL}, "value")],
    [State("color-toggle", "value"),
     State("store-num-data", "data"),
     State("store-class-data", "data"),
     State("store-label-data", "data"),
     State("explanatory-variables", "value"),
     State("target-variables", "value"),
     State("class-variables", "value"),
     State("label-variables", "value"),
     State("regression-method", "value"),
     State("impute-mode", "value"),
           State({"type": "test-table", "index": ALL}, "data"),
      State("store-original-categories", "data"),
      State("store-rare-categories", "data"),
      State("store-original-data", "data")],
    prevent_initial_call=True
)
def update_elastic_net(tabs, log_alpha_list, l1_ratio_list, threshold_list, color_toggle, stored_data, stored_class_data, label_data,  
                       explanatory_variables, target_variables, class_variables, label_variable, regression_method, impute_method, test_data,
                       original_categories, rare_category_store, stored_original_data):
    
    total_variables = target_variables + class_variables
    if stored_data is None or  (not target_variables and not class_variables):
        # データがない場合、各目的変数にデフォルトを返す
        return [
            [{"cv_score": None}] * len(total_variables),
            [html.Div("目的変数が選択されていません。", style={"color": "red"})] * len(total_variables),
            [default_fig("実測値 vs 予測値", "")] * len(total_variables),
            [default_fig("特徴量の重要度", "")] * len(total_variables),
            [default_fig("CV 実測値 vs 予測値", "")] * len(total_variables),
            [[]] * len(total_variables),
            [[{"変数名": "", "回帰係数": "", "p値": "", "選択": False}]] * len(total_variables)
        ]
    elif not explanatory_variables and regression_method in ["elasticnet", "lasso", "ridge"]:
        # 正則化回帰は説明変数必須
        return [
            [{"cv_score": None}] * len(total_variables),
            [html.Div("説明変数が選択されていません。", style={"color": "red"})] * len(total_variables),
            [default_fig("実測値 vs 予測値", "")] * len(total_variables),
            [default_fig("特徴量の重要度", "")] * len(total_variables),
            [default_fig("CV 実測値 vs 予測値", "")] * len(total_variables),
            [[]] * len(total_variables),
            [[{"変数名": "", "回帰係数": "", "p値": "", "選択": False}]] * len(total_variables)
        ]
    
    try:
        data = pd.DataFrame(stored_data["data"])
        raw_df = pd.DataFrame(stored_original_data["data"], columns=stored_original_data["columns"]) if stored_original_data else data.copy()
        cv_X_all = build_cv_feature_frame(raw_df, data, explanatory_variables)
        rare_store_data = rare_category_store.get("data", {}) if rare_category_store else {}
        rare_category_enabled = bool(rare_store_data.get("enabled", False))
        rare_category_map = rare_store_data.get("rare_map", {})
        base_categories_map = rare_store_data.get("base_cats", {})
        class_data = pd.concat([data[explanatory_variables], pd.DataFrame(stored_class_data["data"])], axis=1)
        label_data = pd.DataFrame(label_data["data"]).fillna("データなし")

        # 結果を格納するリスト
        saved_cv_scores = []
        result_messages = []
        scatter_figures = []
        importance_figures = []
        cv_figures = []
        cv_export_data = []
        test_tables = []
        
        if regression_method == "elasticnet":
            method_name = "EN"
        elif regression_method == "lasso":
            method_name = "LASSO"
        elif regression_method == "ridge":
            method_name = "Ridge"
        elif regression_method == "huber":
            method_name = "Huber"
        elif regression_method == "quantile":
            method_name = "QR"

        # 各目的変数に対して処理(回帰)
        for i, target in enumerate(target_variables):
            dropped_data = data.dropna(subset=[target])
            X = dropped_data[explanatory_variables].copy()
            if explanatory_variables:
                preprocessor = NumericPreprocessor(impute_method, scale=True)
                X_scaled = preprocessor.fit_transform(X)
                scaler = preprocessor.scaler
            else:
                X_scaled = X
            y = dropped_data[target].values

            # プロットの色分け用設定
            specific_labels = ["X", "データなし", "O"]  # 分類のラベル順序を定義
            # ラベルの順序と数値割り当てを作成
            if set(label_data[label_variable].unique()).issubset(specific_labels):  # 分類の場合
                label_mapping = {label: idx for idx, label in enumerate(specific_labels)}
            else:
                unique_labels = label_data[label_variable].unique()
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = label_data[label_variable].map(label_mapping)  # 数値ラベルに変換
            numeric_labels = numeric_labels.loc[dropped_data.index]
    
            if color_toggle and label_variable:
                color = numeric_labels  # 数値化されたラベルで色分け
            else:
                color = "blue"  # スイッチが無効の場合はデフォルト色

            hover_label = label_variable if label_variable else "ラベル"
            hovertemplate = (
                f"<b>{hover_label}:</b> "  # ラベル名を含める
                "%{text}<br><b>実測値:</b> %{x}<br><b>予測値:</b> %{y}<extra></extra>"
            )
            
            if regression_method in ["quantile", "huber"]:
                
                X_df = pd.DataFrame(X_scaled, columns=explanatory_variables)
                X_with_const = sm.add_constant(X_df)
                n, p = X_with_const.shape

                if n <= p:
                    saved_cv_scores.append({"target": target, "cv_score": None})
                    result_messages.append(html.Div(f"{method_name}：説明変数の数がサンプル数以上のため、回帰モデルを構築できません。", style={"color": "red"}))
                    scatter_figures.append(default_fig("実測値 vs 予測値", "モデル構築不可"))
                    importance_figures.append(default_fig("特徴量の重要度", "モデル構築不可"))
                    cv_figures.append(default_fig("CV 実測値 vs 予測値", "モデル構築不可"))
                    test_tables.append([{"変数名": "", "回帰係数": "", "p値": "", "選択": False}])
                    continue  # この目的変数の処理をスキップ
                    
                preprocessor = NumericPreprocessor(impute_method, scale=True)
                if regression_method == "quantile":
                    q = log_alpha_list[i] if i < len(log_alpha_list) else 0.5
                    pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="quantile", fit_params={"q": q})
                    pipeline.fit(X, y)
                    y_pred = pipeline.predict(X)
                    result = pipeline.fitted_estimator_
                    standard_errors = result.bse.tolist()
                    p_values = result.pvalues.tolist()
                    msg = html.Div([html.P(f"q: {q:.3f}", style={"color": "#3498db", "fontWeight": "bold"})])
                else:
                    t = log_alpha_list[i] / (1 - log_alpha_list[i]) if i < len(log_alpha_list) else 1.345
                    pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="huber", fit_params={"t": t})
                    pipeline.fit(X, y)
                    y_pred = pipeline.predict(X)
                    result = pipeline.fitted_estimator_
                    # HC3補正済みの共分散行列と個別p値の再計算
                    cov_hc3 = compute_hc3_cov(result)
                    se_hc3 = np.sqrt(np.diag(cov_hc3))
                    t_stats = result.params / se_hc3
                    df_resid = result.df_resid
                    p_values = 2 * t_dist.sf(np.abs(t_stats), df_resid)
                    standard_errors = se_hc3.tolist()
                    p_values = p_values.tolist()
                    if t > 98:
                        msg = html.Div([html.P("t: ∞", style={"color": "#3498db", "fontWeight": "bold"})])
                    else:
                        msg = html.Div([html.P(f"t: {t:.3f}", style={"color": "#3498db", "fontWeight": "bold"})])
                
                scaler = pipeline.preprocessor.scaler
                    
                saved_cv_scores.append({"target": target, "cv_score": None})
                result_messages.append(msg)
                
                variable_names = result.model.exog_names
                coefficients = result.params.tolist()
                # const を (切片) に置き換える
                variable_names = ["(切片)" if v == "const" else v for v in variable_names]
                
                # 散布図
                scatter_fig = go.Figure()
                scatter_fig.add_trace(
                    go.Scatter(
                        x=y,
                        y=y_pred,
                        mode="markers",
                        name="サンプル",
                        marker=dict(color=color, colorscale="Bluered"),
                        text=label_data.loc[dropped_data.index, label_variable] if label_variable else dropped_data.index + "行目",  # ラベル変数がない場合は行番号を表示
                        hovertemplate=hovertemplate
                    )
                )
                scatter_fig = scatter_layout(scatter_fig, y, y_pred, f"{target} ({method_name})")
                scatter_figures.append(scatter_fig)
                
                # 特徴量の重要度
                importance_names = [v for v in variable_names if v != "(切片)"]
                importance_coefs = [c for v, c in zip(variable_names, coefficients) if v != "(切片)"]
                # スケーリングを戻した回帰係数を計算
                orig_coefs = []
                for var, coef in zip(importance_names, importance_coefs):
                    std = scaler.scale_[explanatory_variables.index(var)]
                    orig_coefs.append(coef / std)
                importance_fig = go.Figure()
                importance_fig.add_trace(
                    go.Bar(
                        x=importance_coefs,
                        y=importance_names,
                        orientation="h",
                        customdata=np.array(orig_coefs),  # 元の係数を customdata に渡す
                        hovertemplate="<b>%{y}</b><br>重要度: %{x:.3f}<br>回帰係数: %{customdata:.3f}<extra></extra>"
                    )
                )
                importance_fig.update_layout(title=f"特徴量の重要度 ({method_name})",xaxis_title="重要度",yaxis_title="特徴量")
                importance_figures.append(importance_fig)
                
                #残差プロット
                hover_residual = (
                    f"<b>{hover_label}:</b> "  # ラベル名を含める
                    "%{text}<br><b>予測値:</b> %{x}<br><b>残差:</b> %{y}<extra></extra>"
                )
                
                residuals = y - y_pred
                residual_fig = go.Figure()
                residual_fig.add_trace(go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode="markers",
                    name="サンプル",
                    marker=dict(color=color, colorscale="Bluered"),
                    text=label_data.loc[dropped_data.index, label_variable] if label_variable else dropped_data.index + "行目",  # ラベル変数がない場合は行番号を表示
                    hovertemplate=hover_residual
                ))
                
                residual_fig.update_layout(
                    title=f"残差プロット: {target} ({method_name})",
                    xaxis_title="予測値",
                    yaxis_title="残差",
                    shapes=[dict(type="line", x0=min(y_pred), x1=max(y_pred), y0=0, y1=0,
                                 line=dict(color="gray", dash="dash"))],
                    template="plotly_white"
                )
                
                above_rate = np.mean(y >= y_pred)
                below_rate = np.mean(y <= y_pred)
                
                annotation_text = (
                    f"実測値 ≥ 予測値: {above_rate:.2%}  "
                    f"実測値 ≤ 予測値: {below_rate:.2%}"
                )
                residual_fig.add_annotation(
                    text=annotation_text,
                    xref="paper", yref="paper",
                    x=0.5, y=1.02,
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                cv_figures.append(residual_fig)
                
                # テーブルデータの構築
                table_data = []
                for name, coef, se, pval in zip(variable_names, coefficients, standard_errors, p_values):
                    table_data.append({
                        "変数名": name,
                        "回帰係数": round(coef, 4),
                        "標準誤差": round(se, 4),
                        "p値": round(pval, 4),
                    })
                
                test_tables.append(table_data)
                cv_export_data.append([])
                
            else:
                alpha = 10 ** log_alpha_list[i] if i < len(log_alpha_list) else 1.0
                l1_ratio = l1_ratio_list[i] if i < len(l1_ratio_list) else 0.5
                preprocessor = NumericPreprocessor(impute_method, scale=True)
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
                pipeline = ModelPipeline(preprocessor, model)
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X)
                scaler = pipeline.preprocessor.scaler
    
                # 散布図
                scatter_fig = go.Figure()
                scatter_fig.add_trace(
                    go.Scatter(
                        x=y,
                        y=y_pred,
                        mode="markers",
                        name="サンプル",
                        marker=dict(color=color, colorscale="Bluered"),
                        text=label_data.loc[dropped_data.index, label_variable] if label_variable else dropped_data.index + "行目",  # ラベル変数がない場合は行番号を表示
                        hovertemplate=hovertemplate
                    )
                )
                scatter_fig = scatter_layout(scatter_fig, y, y_pred, f"{target} ({method_name})")
                scatter_figures.append(scatter_fig)
    
                # 特徴量の重要度
                importance_names = explanatory_variables
                importance_coefs = model.coef_.tolist()  # 標準化後の係数（重要度）
                # スケーリングを戻した回帰係数を計算
                orig_coefs = []
                for var, coef in zip(importance_names, importance_coefs):
                    std = scaler.scale_[explanatory_variables.index(var)]
                    orig_coefs.append(coef / std)
                importance_fig = go.Figure()
                importance_fig.add_trace(
                    go.Bar(
                        x=importance_coefs,
                        y=importance_names,
                        orientation="h",
                        customdata=np.array(orig_coefs),  # 元の係数を customdata に渡す
                        hovertemplate="<b>%{y}</b><br>重要度: %{x:.3f}<br>回帰係数: %{customdata:.3f}<extra></extra>"
                    )
                )
                importance_fig.update_layout(
                    title=f"特徴量の重要度 ({method_name})",
                    xaxis_title="重要度",
                    yaxis_title="特徴量",
                    #dragmode="pan"
                )
                importance_figures.append(importance_fig)
    
                # クロスバリデーションプロット
                n_fold = int(np.floor(min(len(y), 10, 5 + 100 / np.sqrt(len(y)))))
                kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)
                all_y_test, all_y_pred = [], []
                cv_fig = go.Figure()
                fold_colors = generate_fold_colors(n_fold)
                
                df_oof = pd.DataFrame(index=data.index, columns=["fold", "y_true", "y_pred"])
                X_cv = cv_X_all.loc[dropped_data.index].copy()
                for fold, (train_idx, test_idx) in enumerate(kf.split(X_cv)):
                    X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    test_index = dropped_data.index[test_idx]

                    preprocessor_fold = NumericPreprocessor(
                        impute_method, scale=True,
                        base_categories=base_categories_map,
                        rare_category_map=rare_category_map,
                        rare_category_enabled=rare_category_enabled
                    )
                    pipeline_fold = ModelPipeline(preprocessor_fold, ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0))
                    pipeline_fold.fit(X_train, y_train)
                    y_pred_fold = pipeline_fold.predict(X_test)
                    fold_features = transformed_feature_frame(preprocessor_fold, X_test, test_index)
                    for feature_name in fold_features.columns:
                        if feature_name not in df_oof.columns:
                            df_oof[feature_name] = np.nan
                        df_oof.loc[test_index, feature_name] = fold_features[feature_name].values
                    all_y_test.extend(y_test)
                    all_y_pred.extend(y_pred_fold)
                    hovertemplate = (
                        f"<b>Fold {fold + 1}</b> "
                        "<br><b>実測値:</b> %{x}<br><b>予測値:</b> %{y}<extra></extra>"
                    )
    
                    cv_fig.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred_fold,
                        mode="markers",
                        name=f"Fold {fold + 1}",
                        marker=dict(color=fold_colors[fold]),
                        hovertemplate=hovertemplate
                    ))
                    
                    df_oof.loc[dropped_data.index[test_idx], "fold"] = fold + 1
                    df_oof.loc[dropped_data.index[test_idx], "y_true"] = y_test.astype(float)
                    df_oof.loc[dropped_data.index[test_idx], "y_pred"] = y_pred_fold.astype(float)
    
                cv_fig = scatter_layout(cv_fig, all_y_test, all_y_pred, f"{n_fold}-fold CV {target} ({method_name})")
                cv_figures.append(cv_fig)
                
                df_oof["行数"] = df_oof.index + 1
                base_cols = ["行数", "fold", "y_true", "y_pred"]
                feature_cols = [c for c in df_oof.columns if c not in base_cols]
                df_oof = df_oof[base_cols + feature_cols]
                cv_export_data.append(df_oof.to_dict("records"))
    
                # 結果メッセージ
                r2 = concordance_corrcoef(all_y_test, all_y_pred)
                mae = mean_absolute_error(all_y_test, all_y_pred)
                saved_cv_scores.append({"target": target, "cv_score": r2})
                if regression_method == "elasticnet":
                    result_message = html.Div([
                        html.P(f"検証データへのモデル適合度: {r2:.5f},　平均誤差: {mae:.3f},　alpha: {alpha:.5f},　l1_ratio: {l1_ratio:.2f}", style={"color": "#3498db", "fontWeight": "bold"})
                    ])
                else:
                    result_message = html.Div([
                        html.P(f"検証データへのモデル適合度: {r2:.5f},　平均誤差: {mae:.3f},　alpha: {alpha:.5f}", style={"color": "#3498db", "fontWeight": "bold"})
                    ])
                result_messages.append(result_message)
                
                test_tables.append([{"変数名": "", "回帰係数": "", "p値": "", "選択": False}])
            
        # 各目的変数に対して処理(分類)
        for i, c_target in enumerate(class_variables):
            log_alpha = log_alpha_list[len(target_variables) + i] if len(log_alpha_list) > len(target_variables) + i else 0
            l1_ratio = l1_ratio_list[len(target_variables) + i] if len(l1_ratio_list) > len(target_variables) + i else 0.5
            dropped_data = class_data.dropna(subset=[c_target]).copy()
            dropped_data.loc[:, c_target] = dropped_data[c_target].map({"O": 1, "X": 0})
            X = dropped_data[explanatory_variables].copy()
            y = dropped_data[c_target].astype(int).values
            threshold = threshold_list[total_variables.index(c_target)]
            preprocessor = NumericPreprocessor(impute_method, scale=True)
            if regression_method in ["quantile", "huber"]: 
                # Firthロジスティック回帰
                method_name = "Firth-LR"
                pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="firth")
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X)
                y_pred_binary = (y_pred > threshold).astype(int)
                result = pipeline.fitted_estimator_
                summary_df = result["summary"]
                coefficients = summary_df["回帰係数"].values
                standard_errors = summary_df["標準誤差"].values
                p_values = summary_df["p値"].values
                variable_names = summary_df["変数名"].values
                unique_classes = np.unique(y)
                
            else:
                # 通常のElasticNet分類（正則化あり）
                alpha = 10 ** log_alpha
                model = LogisticRegression(
                            penalty="elasticnet", 
                            solver="saga", 
                            C=1/alpha, 
                            l1_ratio=l1_ratio, 
                            class_weight='balanced', 
                            random_state=0
                        )
                pipeline = ModelPipeline(preprocessor, model)
                pipeline.fit(X, y)
                y_pred = pipeline.predict_proba(X)[:, 1]
                y_pred_binary = [1 if prob > threshold else 0 for prob in y_pred]
                unique_classes = model.classes_
                coefficients = model.coef_.flatten()
                variable_names = explanatory_variables
                
            # 特徴量の重要度
            importance_fig = go.Figure()
            importance_fig.add_trace(
                go.Bar(
                    x=coefficients, 
                    y=variable_names, 
                    orientation="h", 
                    hovertemplate="<b>%{y}</b><br>重要度: %{x:.3f}<extra></extra>"
                )
            )
            importance_fig.update_layout(
                title=f"特徴量の重要度 ({method_name})",
                xaxis_title="重要度",yaxis_title="特徴量",
                #dragmode="pan"
            )
            importance_figures.append(importance_fig)

            # プロットの色分け用設定
            specific_labels = ["X", "データなし", "O"]  # 分類のラベル順序を定義
            # ラベルの順序と数値割り当てを作成
            if set(label_data[label_variable].unique()).issubset(specific_labels):  # 分類の場合
                label_mapping = {label: idx for idx, label in enumerate(specific_labels)}
            else:
                unique_labels = label_data[label_variable].unique()
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = label_data[label_variable].map(label_mapping)  # 数値ラベルに変換
            numeric_labels = numeric_labels.loc[dropped_data.index]
    
            if color_toggle and label_variable:
                color = numeric_labels  # 数値化されたラベルで色分け
            else:
                color = "blue"  # スイッチが無効の場合はデフォルト色
                
            hover_label = label_variable if label_variable else "ラベル"
            hovertemplate = (
                f"<b>{hover_label}:</b> "  # ラベル名を含める
                "%{text}<br><b>実測値:</b> %{x}<br><b>予測陽性確率:</b> %{y}<extra></extra>"
            )      

            # 散布図: 実測値 (x軸) vs 予測確率 (y軸)
            sigmoid_fig = go.Figure()
            
            jitter_width = 0.2     # 実測値 y に少しランダムノイズを追加
            np.random.seed(0)
            y_jitter = y + np.random.uniform(-jitter_width, jitter_width, size=len(y))
            
            sigmoid_fig.add_trace(
                go.Scatter(
                    x=y_jitter,  # 実測値
                    y=y_pred,  # 予測確率
                    mode="markers",
                    name="全サンプル",
                    marker=dict(color=color, colorscale="Bluered"),
                    text=label_data.loc[dropped_data.index, label_variable] if label_variable else dropped_data.index + "行目",
                    hovertemplate=hovertemplate,
                )
            )
            
            # 閾値の線
            sigmoid_fig.add_shape(
                type="line",
                x0=-0.5, x1=4.5,  # 実測値の範囲に合わせる
                y0=threshold, y1=threshold,
                line=dict(color="red", width=2, dash="dash"),  # スタイル調整可
                name="閾値"
            )

            # レイアウトの設定
            if regression_method in ["quantile", "huber"]:
                sigmoid_fig.update_layout(
                    title=f"{c_target}の実測値 vs 予測確率 ({method_name})",
                    #dragmode="pan",
                    xaxis=dict(title="実測値", tickvals=[0, 1], ticktext=["X", "O"], range=[-0.5, 1.5], fixedrange=False),
                    yaxis=dict(title="予測値", range=[-0.1, 1.1], fixedrange=False)
                )
            else:
                sigmoid_fig.update_layout(
                    title=f"{c_target}の実測値 vs 予測確率 ({method_name})",
                    #dragmode="pan",
                    xaxis=dict(title="実測値", tickvals=[0, 1, 3, 4], ticktext=["X", "O", "X", "O"], range=[-0.5, 4.5], fixedrange=False),
                    yaxis=dict(title="予測値", range=[-0.1, 1.1], fixedrange=False)
                )
        
            cv_figures.append(sigmoid_fig)

            # OOFデータ出力用
            if regression_method in ["quantile", "huber"]:
                cv_export_data.append([])
            else:
                df_oof = pd.DataFrame(index=data.index, columns=["fold", "y_true", "y_pred"])
                X_cv = cv_X_all.loc[dropped_data.index].copy()
                min_class_count = np.min(np.bincount(y))
                n_fold = int(np.floor(min(min_class_count, 10, 5 + 100 / np.sqrt(len(y)))))
                kf = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=1, random_state=0)
                fold_colors = generate_fold_colors(n_fold)
                all_y_test, all_y_pred, all_y_proba = [], [], []
                for fold, (train_idx, test_idx) in enumerate(kf.split(X_cv, y)):
                    X_train, X_test = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    test_index = dropped_data.index[test_idx]
                
                    preprocessor_fold = NumericPreprocessor(
                        impute_method, scale=True,
                        base_categories=base_categories_map,
                        rare_category_map=rare_category_map,
                        rare_category_enabled=rare_category_enabled
                    )
                    pipeline_fold = ModelPipeline(preprocessor_fold, model)
                    pipeline_fold.fit(X_train, y_train)
                    cv_pred_proba = pipeline_fold.predict_proba(X_test)[:, 1]  # positive class の確率
                    fold_features = transformed_feature_frame(preprocessor_fold, X_test, test_index)
                    for feature_name in fold_features.columns:
                        if feature_name not in df_oof.columns:
                            df_oof[feature_name] = np.nan
                        df_oof.loc[test_index, feature_name] = fold_features[feature_name].values
                    cv_pred_binary = (cv_pred_proba >= threshold).astype(int)
                    
                    y_labels = ["O" if val == 1 else "X" for val in y_test]
                    
                    hovertemplate = (
                        f"<b>Fold {fold + 1}</b>"
                        "<br><b>実測値:</b> %{customdata}<br><b>予測陽性確率:</b> %{y}<extra></extra>"
                    )
                    
                    sigmoid_fig.add_trace(go.Scatter(
                        x=3 + y_test + np.random.uniform(-jitter_width, jitter_width, size=len(y_test)),
                        y=cv_pred_proba,
                        customdata=y_labels,
                        mode="markers",
                        name=f"Fold {fold + 1}",
                        marker=dict(color=fold_colors[fold]),
                        hovertemplate=hovertemplate
                    ))
                    
                    all_y_test.extend(y_test)
                    all_y_pred.extend(cv_pred_binary)
                    all_y_proba.extend(cv_pred_proba) 
                
                    # fold の結果を OOF に追加
                    df_oof.loc[dropped_data.index[test_idx], "fold"] = fold + 1
                    df_oof.loc[dropped_data.index[test_idx], "y_true"] = y_test.astype(int)
                    df_oof.loc[dropped_data.index[test_idx], "y_pred"] = cv_pred_proba.astype(float)
                
                # DataFrame 化して export 用に保存
                df_oof["行数"] = df_oof.index + 1
                base_cols = ["行数", "fold", "y_true", "y_pred"]
                feature_cols = [c for c in df_oof.columns if c not in base_cols]
                df_oof = df_oof[base_cols + feature_cols]
                cv_export_data.append(df_oof.to_dict("records"))
            
            # サブプロット作成
            cm_figure = sp.make_subplots(
                            rows=2, cols=1,
                            subplot_titles=[f"{c_target}の混同行列", "評価指標"],
                            row_heights=[0.7, 0.3],
                            specs=[[{"type": "heatmap"}], [{"type": "table"}]]
                        )

            # 混同行列
            if regression_method in ["quantile", "huber"]:
                cm = confusion_matrix(y, y_pred_binary, labels=unique_classes).T
            else:
                cm = confusion_matrix(all_y_test, all_y_pred, labels=unique_classes).T
            class_labels = ["X" if cl == 0 else "O" for cl in unique_classes]
            
            cm_figure.add_trace(
                go.Heatmap(
                    z=cm,
                    x=class_labels,
                    y=class_labels,
                    text=cm,
                    texttemplate="%{text}",
                    colorscale=[[0, 'white'], [1, 'red']],
                    zmin=0,
                    zmax=len(y),
                    showscale=False,
                    hovertemplate=(
                        "実測値: %{x}<br>"
                        "予測値: %{y}<br>"
                        "値 (カウント): %{z}<extra></extra>"
                    )
                ),
                row=1, col=1
            )
            
            # --- テーブル用データ作成 ---
            metric_names = [
                "Accuracy (正しく分類できた割合)",
                "Precision (陽性と分類して実際に陽性だった割合)",
                "Recall (実際の陽性を陽性と分類できた割合)",
                "F1 Score (PrecisionとRecallの両立度合い)"
            ]
            
            metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score"]
            
            # --- 学習データの指標 ---
            train_metrics = {
                "Accuracy": accuracy_score(y, y_pred_binary),
                "Precision": precision_score(y, y_pred_binary),
                "Recall": recall_score(y, y_pred_binary),
                "F1 Score": f1_score(y, y_pred_binary)
            }
            
            if regression_method in ["quantile", "huber"]:
                metric_values = [
                    metric_names,
                    [train_metrics[k] for k in metric_keys],   # 学習
                ]
                
                # --- テーブル追加 ---
                cm_figure.add_trace(
                    go.Table(
                        columnwidth=[260, 70],
                        header=dict(
                            values=["指標", "学習"],
                            fill_color="lightgray",
                            align="center"
                        ),
                        cells=dict(
                            values=metric_values,
                            align="center",
                            format=[None, ".4f"]
                        )
                    ),
                    row=2, col=1
                )
                r2 = matthews_corrcoef(y, y_pred_binary)
                mae = mean_absolute_error(y, y_pred) * 100
            else:
                # --- 検証データの指標 ---
                valid_metrics = {
                    "Accuracy": accuracy_score(all_y_test, all_y_pred),
                    "Precision": precision_score(all_y_test, all_y_pred),
                    "Recall": recall_score(all_y_test, all_y_pred),
                    "F1 Score": f1_score(all_y_test, all_y_pred)
                }
            
                metric_values = [
                    metric_names,
                    [train_metrics[k] for k in metric_keys],   # 学習
                    [valid_metrics[k] for k in metric_keys]    # 検証
                ]
            
                # --- テーブル追加 ---
                cm_figure.add_trace(
                    go.Table(
                        columnwidth=[260, 70, 70],
                        header=dict(
                            values=["指標", "学習", "検証"],
                            fill_color="lightgray",
                            align="center"
                        ),
                        cells=dict(
                            values=metric_values,
                            align="center",
                            format=[None, ".4f", ".4f"]
                        )
                    ),
                    row=2, col=1
                )
                r2 = matthews_corrcoef(all_y_test, all_y_pred)
                mae = mean_absolute_error(all_y_test, all_y_proba) * 100
            
            cm_figure.update_layout(
                title=f"{c_target}の分類結果 ({method_name})",
                xaxis=dict(
                    title="実測値",
                    tickmode="array",
                    tickvals=class_labels,
                    ticktext=class_labels,  # クラスラベル ("X" と "O")
                    scaleanchor="y",
                    constrain="domain"
                ),
                yaxis=dict(
                    title="予測値",
                    tickmode="array",
                    tickvals=class_labels,
                    ticktext=class_labels  # クラスラベル ("X" と "O")
                ),
                dragmode=False
            )
            scatter_figures.append(cm_figure)

            # 結果メッセージ
            saved_cv_scores.append({"target": c_target, "cv_score": r2})
            if regression_method in ["quantile", "huber"]:
                result_message = html.Div([
                    html.P(f"学習データへの適合度: {r2:.5f},　平均誤差: {mae:.3f}%", style={"color": "#3498db", "fontWeight": "bold"})
                ])
            else:
                result_message = html.Div([
                    html.P(f"検証データへの適合度: {r2:.5f},　平均誤差: {mae:.3f}%,　alpha: {alpha:.5f},　l1_ratio: {l1_ratio:.2f}", style={"color": "#3498db", "fontWeight": "bold"})
                ])
            result_messages.append(result_message)
            
            table_data = []
            if regression_method in ["quantile", "huber"]:
                for name, coef, se, pval in zip(variable_names, coefficients, standard_errors, p_values):
                    table_data.append({
                        "変数名": name,
                        "回帰係数": round(coef, 4),
                        "標準誤差": round(se, 4) if se is not None else "",
                        "p値": round(pval, 4) if pval is not None else "",
                        "選択": False
                    })
                test_tables.append(table_data)
            else:
                test_tables.append([{"変数名": "", "回帰係数": "", "p値": "", "選択": False}])

        return [
            saved_cv_scores,
            result_messages,
            scatter_figures,
            importance_figures,
            cv_figures,
            cv_export_data,
            test_tables
        ]
    except Exception as e:
        # エラー時の処理
        return [
            [{"cv_score": None}] * len(total_variables),
            [html.Div(f"解析中にエラーが発生しました: {str(e)}", style={"color": "red"})] * len(total_variables),
            #[html.Div("解析中...", style={"color": "red"})] * len(total_variables),
            [default_fig("実測値 vs 予測値", "")] * len(total_variables),
            [default_fig("特徴量の重要度", "")] * len(total_variables),
            [default_fig("CV 実測値 vs 予測値", "")] * len(total_variables),
            [[]] * len(total_variables),
            [[{"変数名": "", "回帰係数": "", "p値": "", "選択": False}]] * len(total_variables)
        ]

# Wald検定
@app.callback(
    Output({"type": "wald-score", "index": MATCH}, "children"),
    [Input({"type": "test-table", "index": MATCH}, "selected_rows"),
     Input({"type": "alpha-slider", "index": MATCH}, "value")],
    [State({"type": "test-table", "index": MATCH}, "data"),
     State("store-num-data", "data"),
     State("store-class-data", "data"),
     State("explanatory-variables", "value"),
     State("regression-method", "value"),
     State({"type": "test-table", "index": MATCH}, "id"),
     State("impute-mode", "value")]
)

def run_wald_test(selected_rows, alpha, table_data, stored_data, stored_class_data, 
                  explanatory_variables, regression_method, table_id, impute_method):
    if not selected_rows:
        return html.Div("行を(複数)選択で同時検定", style={"color": "gray"})

    selected_variables = [table_data[i]["変数名"] for i in selected_rows]
    target_variable = table_id["index"]
    
    df_reg = pd.DataFrame(stored_data["data"])
    df_clf = pd.DataFrame(stored_class_data["data"])

    if target_variable in df_reg.columns:
        task_type = "regression"
        df = df_reg.dropna(subset=[target_variable])
        y = df[target_variable].values
    elif target_variable in df_clf.columns:
        task_type = "classification"
        class_data = pd.concat([df_reg[explanatory_variables], df_clf], axis=1)
        df = class_data.dropna(subset=[target_variable]).copy()
        df[target_variable] = df[target_variable].map({"O": 1, "X": 0})
        y = df[target_variable].values
        y = y.astype(int)
    else:
        raise ValueError(f"{target_variable} がどちらのデータにも存在しません")

    X = df[explanatory_variables].copy()
    preprocessor = NumericPreprocessor(impute_method, scale=True)
    X_scaled = preprocessor.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=explanatory_variables)
    X_with_const = sm.add_constant(X_df)

    if task_type == "regression":
        if regression_method == "quantile":
            model = sm.QuantReg(y, X_with_const)
            result = model.fit(q=alpha, method="bootstrap")
        else:
            t = alpha / (1 - alpha)
            model = sm.RLM(y, X_with_const, M=sm.robust.norms.HuberT(t=t))
            result = model.fit()
            cov_hc3 = compute_hc3_cov(result)
            beta = result.params
            
        # Wald検定の構築
        joint_indices = [j for j, name in enumerate(result.model.exog_names) if name in selected_variables]
        R = np.zeros((len(joint_indices), len(result.params)))
        for j, idx in enumerate(joint_indices):
            R[j, idx] = 1
                
        if regression_method == "quantile":
            wald_result = result.wald_test(R, scalar=True)
            wald_pval = round(float(wald_result.pvalue), 4)
        else:
            if len(joint_indices) == 1:
                j = joint_indices[0]
                se = np.sqrt(cov_hc3[j, j])
                t_stat = beta[j] / se
                df_resid = result.df_resid
                wald_pval = round(2 * t_dist.sf(abs(t_stat), df_resid), 4)
            else:
                R_beta = R @ beta
                R_cov = R @ cov_hc3 @ R.T
                try:
                    inv_cov = np.linalg.inv(R_cov)
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.pinv(R_cov)
                wald_stat = float(R_beta.T @ inv_cov @ R_beta)
                df_wald = len(joint_indices)
                wald_pval = round(chi2.sf(wald_stat, df_wald), 4)
    else:
        firth_result = firth_logistic_regression(X_df, y, False, True)
        beta = pd.Series(firth_result["coefficients"], index=["Intercept"] + explanatory_variables)
        cov_matrix = pd.DataFrame(np.diag(firth_result["standard_errors"]**2), columns=beta.index, index=beta.index)
        
        joint_indices = [j for j, name in enumerate(beta.index) if name in selected_variables]
        R = np.zeros((len(joint_indices), len(beta)))
        for j, idx in enumerate(joint_indices):
            R[j, idx] = 1
    
        R_beta = R @ beta.values
        R_cov = R @ cov_matrix.values @ R.T
        try:
            inv_cov = np.linalg.inv(R_cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(R_cov)
        wald_stat = float(R_beta.T @ inv_cov @ R_beta)
        df_wald = len(joint_indices)
        wald_pval = round(chi2.sf(wald_stat, df_wald), 4)
        
    return html.Div([
        html.Div(f"選択された {len(selected_variables)} 変数に対する同時検定"),
        html.Div(f"p値: {wald_pval}", style={"color": "green"})
    ])

# コールバック: 説明変数ごとにスライダー/ドロップダウン作成
@app.callback(
    Output("input-fields", "children"),
    Input("analyze-button", "n_clicks"),
    [State("explanatory-variables", "value"),
     State("store-num-data", "data"),
     State("store-original-categories", "data"),
     State("store-rare-categories", "data")]
)
def generate_sliders(n_clicks, explanatory_variables, stored_data, original_categories, rare_category_map):
    if not explanatory_variables or stored_data is None:
        return html.Div("説明変数が選択されていません。", style={"color": "red"})

    data = pd.DataFrame(stored_data["data"])
    fields = []
    grouped = group_encoded_columns(explanatory_variables)  #グループ化

    processed_categories = {}

    for col, cats in original_categories["data"].items():
        rare_cats = rare_category_map["data"].get(col, [])
    
        if len(rare_cats) >= 2:
            # レアカテゴリを除外して「その他」を追加
            new_cats = [c for c in cats if c not in rare_cats]
            new_cats.append("その他")
            processed_categories[col] = new_cats
        else:
            # 統合されていないので元のカテゴリのまま
            processed_categories[col] = cats
            
    for base, cols in grouped.items():
        if base in processed_categories:
            # --- カテゴリ変数（One-hotエンコードされたもの） ---
            categories = processed_categories[base]
            options = [{"label": cat, "value": cat} for cat in categories]

            # ドロップダウン（ユーザーに見せる）
            fields.append(
                html.Div([
                    html.Label(f"{base}:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id={"type": "category-dropdown", "index": base},
                        options=options,
                        value=categories[0],
                        clearable=False,
                        style={"width": "70%"}
                    )
                ], style={"marginBottom": "20px"})
            )

            # 元のスライダー（制御用に残すが非表示）
            for col in cols:
                min_val = data[col].min() if not data[col].isnull().all() else 0
                max_val = data[col].max() if not data[col].isnull().all() else 1
                fields.append(
                    html.Div(
                        dcc.Slider(
                            id={"type": "range-slider", "index": col},
                            min=min_val,
                            max=max_val,
                            value=float(min_val),
                            step=None,  # 中間値を選べないようにする
                            marks={float(min_val): "X", float(max_val): "O"},
                        ),
                        style={"width": "40%"} #"display": "none"}  # 非表示にする
                    )
                )
        else:
            # --- 数値変数 ---
            col = cols[0]
            min_val = data[col].min() if not data[col].isnull().all() else 0
            max_val = data[col].max() if not data[col].isnull().all() else 1

            fields.append(
                html.Div([
                    html.Label(f"{col}:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                    html.Div([
                        dcc.Input(
                            id={"type": "min-input", "index": col},
                            type="number",
                            value=float(min_val),
                            style={"width": "15%", "marginRight": "10px", "fontSize": "12px"}
                        ),
                        html.Div(
                            dcc.Slider(
                                id={"type": "range-slider", "index": col},
                                min=float(min_val),
                                max=float(max_val),
                                value=float(min_val),
                                marks=None,
                                tooltip={"always_visible": True, "placement": "bottom"}
                            ),
                            style={"width": "65%", "marginRight": "10px"}
                        ),
                        dcc.Input(
                            id={"type": "max-input", "index": col},
                            type="number",
                            value=float(max_val),
                            style={"width": "15%", "marginRight": "10px", "fontSize": "12px"}
                        )
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})
                ], style={"marginBottom": "20px"})
            )

    return fields

# コールバック: スライダーの範囲更新
@app.callback(
    [Output({"type": "range-slider", "index": MATCH}, "min"),
     Output({"type": "range-slider", "index": MATCH}, "max")],
    [Input({"type": "min-input", "index": MATCH}, "value"),
     Input({"type": "max-input", "index": MATCH}, "value")]
)
def update_slider_range(min_value, max_value):
    if min_value is None or max_value is None or min_value >= max_value:
        return no_update, no_update
    return min_value, max_value

# コールバック: ドロップダウンに応じて非表示スライダー更新
@app.callback(
    Output({"type": "range-slider", "index": ALL}, "value"),
    Input({"type": "category-dropdown", "index": ALL}, "value"),
    [State({"type": "category-dropdown", "index": ALL}, "id"),
     State({"type": "range-slider", "index": ALL}, "id"),
     State("store-num-data", "data"),
     State("explanatory-variables", "value")]
)
def update_hidden_sliders(selected_categories, dropdown_ids, slider_ids, stored_data, explanatory_variables):

    data = pd.DataFrame(stored_data["data"])
    grouped = group_encoded_columns(explanatory_variables)

    # UI で選ばれたカテゴリを辞書化
    selected_map = {
        dd_id["index"]: sel
        for sel, dd_id in zip(selected_categories, dropdown_ids)
    }

    results = []

    # スライダーの順番でループ
    for sid in slider_ids:
        col = sid["index"]

        # --- 数値スライダーは no_update ---
        if "[" not in col:
            results.append(no_update)
            continue

        # --- カテゴリスライダー ---
        base = col.split("[")[0]
        cat = col[len(base)+1:-1]

        selected = selected_map.get(base, None)

        # sum-to-zero / one-hot 共通で min/max を使う
        min_val = data[col].min() if not data[col].isnull().all() else 0
        max_val = data[col].max() if not data[col].isnull().all() else 1

        # cols に存在しないカテゴリ = drop-first の基準カテゴリ
        if selected is not None and selected not in [c[len(base)+1:-1] for c in grouped[base]]:
            results.append(min_val)
        else:
            results.append(max_val if cat == selected else min_val)

    return results

# コールバック: 予測値計算
@app.callback(
    [Output("prediction-results", "children"),
     Output("store-scalers", "data")],
    [Input("tabs", "value"),
     Input({"type": "range-slider", "index": ALL}, "value")],
    [State("explanatory-variables", "value"),
     State("target-variables", "value"),
     State("class-variables", "value"),
     State("store-num-data", "data"),
     State("store-class-data", "data"),
     State({"type": "alpha-slider", "index": ALL}, "value"),
     State({"type": "l1-ratio-slider", "index": ALL}, "value"),
     State({"type": "threshold-slider", "index": ALL}, "value"),
     State({"type": "saved-cv-scores", "index": ALL}, "data"),
     State("regression-method", "value"),
     State("impute-mode", "value")]
)
def update_predictions(selected_tab, slider_values, explanatory_variables, target_variables, class_variables, 
                       stored_data, stored_class_data, alpha_values, l1_ratio_values, threshold_values,
                       saved_cv_scores, regression_method, impute_method):
    
    if selected_tab != "prediction-tab":
        return no_update, no_update  # 別タブなら何もしない
    
    if not slider_values or saved_cv_scores is None:
        return html.Div("解析終了後、スライダーを操作して結果を表示", style={"color": "red"}), no_update

    try:
        # スライダー値を使った入力データを生成
        input_data = pd.DataFrame([slider_values], columns=explanatory_variables)
        data = pd.DataFrame(stored_data["data"])
        class_data = pd.concat([data[explanatory_variables], pd.DataFrame(stored_class_data["data"])], axis=1)

        # サブプロット作成（目的変数の数だけ列を作成）
        total_variables = target_variables + class_variables
        fig = sp.make_subplots(
            rows=1, cols=len(total_variables),
            shared_yaxes=False,  # y軸はそれぞれ独立
        )
        
        scaler_list = []

        for i, (target, alpha, l1_ratio, score) in enumerate(zip(target_variables, alpha_values, l1_ratio_values, saved_cv_scores)):
            # 回帰モデルを構築して予測値を計算
            dropped_data = data.dropna(subset=[target])
            X = dropped_data[explanatory_variables].copy()
            y = dropped_data[target].values
            
            preprocessor = NumericPreprocessor(impute_method, scale=True)
            if regression_method == "huber":
                t = alpha / (1 - alpha)
                pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="huber", fit_params={"t": t})
            elif regression_method == "quantile":
                q = alpha
                pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="quantile", fit_params={"q": q})
            else:
                model = ElasticNet(alpha=(10 ** alpha), l1_ratio=l1_ratio, random_state=0)
                pipeline = ModelPipeline(preprocessor, model)
            
            pipeline.fit(X, y)
            prediction = pipeline.predict(input_data)[0]
            
            scaler = pipeline.preprocessor.scaler
            scaler_list.append({"center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()})

            # 箱ひげ図をサブプロットに追加
            fig.add_trace(
                go.Box(
                    y=y,
                    name=target,
                    boxpoints=False,
                    fillcolor="rgba(0, 0, 255, 0.2)",
                    line=dict(color="rgba(0, 0, 255, 0.5)")
                ),
                row=1, col=i + 1
            )

            # 予測値を箱ひげ図の上にプロット
            fig.add_trace(
                go.Scatter(
                    x=[target],
                    y=[prediction],
                    mode="markers+text",
                    text=[f"{prediction:.2f}"],
                    textfont=dict(color="red", size=14),
                    textposition="top center",
                    marker=dict(color="red", size=10),
                    name="予測値"
                ),
                row=1, col=i + 1
            )
            if regression_method in ["quantile", "huber"]:
                pass
            else:
                # x軸タイトルにCVのR²スコアを表示
                fig.update_xaxes(
                    title=dict(
                        text=f"適合度: {score['cv_score']:.3f}",
                        font=dict(size=12, color="darkblue")
                    ),
                    row=1, col=i + 1
                )

        for j, (c_target, alpha, l1_ratio, score) in enumerate(zip(class_variables, alpha_values, l1_ratio_values, saved_cv_scores[len(target_variables):])):
            # 分類モデルを構築して陽性確率を計算
            dropped_data = class_data.dropna(subset=[c_target]).copy()
            dropped_data.loc[:, c_target] = dropped_data[c_target].map({"O": 1, "X": 0})
            X = dropped_data[explanatory_variables].copy()
            y = dropped_data[c_target].astype(int).values
            
            preprocessor = NumericPreprocessor(impute_method, scale=True)
            if regression_method in ["quantile", "huber"]:
                pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="firth")
                pipeline.fit(X, y)
                prediction_prob = pipeline.predict_proba(input_data)[0]
            else:
                model = LogisticRegression(
                    penalty="elasticnet", solver="saga", C=1/(10 ** alpha), l1_ratio=l1_ratio, random_state=0
                )
                pipeline = ModelPipeline(preprocessor, model)
                pipeline.fit(X, y)
                prediction_prob = pipeline.predict_proba(input_data)[0, 1]  # クラス1の確率
            
            scaler = pipeline.preprocessor.scaler
            scaler_list.append({"center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()})

            # 確率をサブプロットに追加（単純な棒グラフ）
            threshold_idx = len(target_variables) + j
            if threshold_idx < len(threshold_values)and threshold_values[threshold_idx] is not None:
                threshold = threshold_values[threshold_idx]
            else:
                threshold = 0.5
                
            if prediction_prob >= threshold:
                bar_color = "rgba(255, 0, 0, 0.3)"
                text_color = "rgb(255, 0, 0)"
                text_OX = "O"
            else:
                bar_color = "rgba(0, 0, 255, 0.3)"
                text_color = "rgb(0, 0, 255)"
                text_OX = "X"
                
            fig.add_trace(
                go.Bar(
                    x=[c_target],
                    y=[prediction_prob],
                    text=[f"{prediction_prob:.2%} → {text_OX}"],
                    textfont=dict(color=text_color, size=14),
                    textposition="outside",
                    marker=dict(color=bar_color),
                    name="陽性確率"
                ),
                row=1, col=len(target_variables) + j + 1
            )

            if regression_method in ["quantile", "huber"]:
                pass
            else:
                # x軸タイトルにCVのAUCスコアを表示
                fig.update_xaxes(
                    title=dict(
                        text=f"適合度: {score['cv_score']:.3f}",
                        font=dict(size=12, color="darkblue")
                    ),
                    row=1, col=len(target_variables) + j + 1
                )
            fig.update_yaxes(
                range=[0, 1.1],  # y 軸の範囲を [0, 1.1] に設定
                row=1, col=len(target_variables) + j + 1
            )

        if regression_method == "elasticnet":
            method_name = "EN"
        elif regression_method == "lasso":
            method_name = "LASSO"
        elif regression_method == "ridge":
            method_name = "Ridge"
        elif regression_method == "huber":
            method_name = "Huber/Firth-LR"
        else:
            method_name = "QR/Firth-LR"

        # レイアウト設定
        fig.update_layout(
            title=f"目的変数の分布と予測値 ({method_name})",
            showlegend=False,
            plot_bgcolor="#f9f9f9",
        )

        # 一括予測アップロード部分
        bulk_predict = html.Div(
            style={"marginTop": "30px"},
            children=[
                html.Label("一括予測:", style={"marginRight": "10px", "fontWeight": "bold"}),
        
                dcc.Upload(
                    id="bulk-predict-upload",
                    children=html.Button("アップロード", style={"marginLeft": "10px"}),
                    multiple=False,
                    accept=".csv",
                    style={"display": "inline-block"}
                ),
        
                html.Span(id="bulk-predict-status",
                          style={"marginLeft": "15px", "fontWeight": "bold"}),
        
                dcc.Download(id="bulk-predict-download")
            ]
        )

        return html.Div(
                    children=[
                        dcc.Graph(figure=fig),
                        bulk_predict
                    ]
                ), scaler_list

    except Exception as e:
        return html.Div(f"エラーが発生しました: {str(e)}", style={"color": "red"}), no_update
        #return html.Div("解析終了後、スライダーを操作して結果を表示", style={"color": "red"}), no_update

@app.callback(
    [Output("bulk-predict-status", "children"),
     Output("bulk-predict-status", "style"),
     Output("bulk-predict-download", "data")],
    Input("bulk-predict-upload", "contents"),
    State("bulk-predict-upload", "filename"),
    State("explanatory-variables", "value"),
    State("target-variables", "value"),
    State("class-variables", "value"),
    State({"type": "dynamic-dropdown", "col": ALL}, "value"),
    State({"type": "dynamic-dropdown", "col": ALL}, "id"),
    State("impute-mode", "value"),
    State({"type": "alpha-slider", "index": ALL}, "value"),
    State({"type": "l1-ratio-slider", "index": ALL}, "value"),
    State({"type": "threshold-slider", "index": ALL}, "value"),
    State({"type": "saved-cv-scores", "index": ALL}, "data"),
    State("regression-method", "value"),
    State("store-num-data", "data"),
    State("store-class-data", "data"),
    State("store-label-data", "data"),
    State("store-scalers", "data"),
    State("store-rare-categories", "data"), # ★学習時のOne-hot後の列リストが必要
    prevent_initial_call=True
)
def bulk_predict(contents, filename, explanatory_vars, target_variables, class_variables,
                 dropdown_values, dropdown_ids, impute_mode, alpha_values, l1_ratio_values, 
                 threshold_values, saved_cv_scores, regression_method,
                 stored_num_data, stored_class_data, label_data, stored_scalers, rare_category_store):

    if contents is None:
        return "", {}, None

    # -------------------------
    # ① CSV 読み込み
    # -------------------------
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    result = chardet.detect(decoded)
    encoding = result["encoding"] if result["encoding"] else "utf-8"

    try:
        df_new = pd.read_csv(io.StringIO(decoded.decode(encoding)))
    except:
        df_new = pd.read_csv(io.StringIO(decoded.decode("cp932")))

    # 「%」付き表記を数値に変換（例: 10% -> 0.1）
    df_new = convert_percent_columns(df_new)

    # -------------------------
    # ② 説明変数チェック
    # -------------------------
    selected_explanatory = sorted({col.split("[")[0] for col in explanatory_vars})
    missing = [col for col in selected_explanatory if col not in df_new.columns]
    if missing:
        return f"不足している変数: {', '.join(missing)}", {"color": "red"}, None
    
    # -------------------------
    # ③ 説明変数・目的変数のエンコード（未知カテゴリの処理対応）
    # -------------------------
    rare_category_map = {}
    if rare_category_store and "data" in rare_category_store:
        rare_category_map = rare_category_store["data"]

    # 学習時のカテゴリ一覧を取得
    train_categories_map = {}
    for col in selected_explanatory:
        cats = [c[len(col)+1:-1] for c in explanatory_vars if c.startswith(f"{col}[") and c.endswith("]")]
        if cats:
            train_categories_map[col] = set(cats)

    df_proc = df_new.copy()
    
    # レアカテゴリ置換および未知カテゴリの処理
    for col in selected_explanatory:
        if col not in df_proc.columns or pd.api.types.is_numeric_dtype(df_proc[col]):
            continue

        rare_cats = set(rare_category_map.get(col, []))
        known_dummy_cats = train_categories_map.get(col, set())
        has_other = "その他" in known_dummy_cats or (len(rare_cats) >= 2)

        def map_category(x):
            if pd.isna(x) or x is None or x == "":
                return "欠損" if impute_mode in ["iterative_missing", "missing"] else np.nan
            s = str(x)
            # レアカテゴリの統合
            if len(rare_cats) >= 2 and s in rare_cats:
                return "その他"
            # 未知カテゴリの判定
            # (known_dummy_cats に含まれるか、またはdropされた基準カテゴリかチェック)
            # 訓練時の既知カテゴリでなければ未知
            # もし「その他」が存在するなら「その他」にマッピング
            # 存在しなければそのまま残す（ダミー変数作成時に一致する列が無く全0となる）
            return s

        df_proc[col] = df_proc[col].map(map_category)

    num_data_base = pd.DataFrame()
    
    # ラベルデータ（目的変数候補）
    df_category = pd.DataFrame(label_data["data"])
    df_category.columns = label_data["columns"]
    
    # --- 元の構造を維持したループ ---    
    for col in selected_explanatory:
        if col not in df_proc.columns:
            continue
    
        # 数値列
        if pd.api.types.is_numeric_dtype(df_proc[col]):
            num_data_base[col] = df_proc[col]
            continue
    
        # カテゴリ列
        rare_cats = set(rare_category_map.get(col, []))
        known_dummy_cats = train_categories_map.get(col, set())
        has_other = "その他" in known_dummy_cats or (len(rare_cats) >= 2)

        # 未知カテゴリの置換判定
        # 学習時の全カテゴリ（基準カテゴリ含む）
        train_cats_with_base = set(known_dummy_cats)
        if "その他" in known_dummy_cats:
            train_cats_with_base.add("その他")
        # rare_cats も学習時に存在したもの
        train_cats_with_base.update(rare_cats)

        def process_cat_val(val):
            if pd.isna(val) or val is None or val == "":
                return "欠損" if impute_mode in ["iterative_missing", "missing"] else np.nan
            s = str(val)
            if len(rare_cats) >= 2 and s in rare_cats:
                return "その他"
            # 訓練時に存在しない未知カテゴリ
            if s not in train_cats_with_base and s != "その他":
                if has_other:
                    return "その他"
                else:
                    return "__UNKNOWN__"
            return s

        series_col = df_proc[col].map(process_cat_val)

        dummies = pd.get_dummies(
            series_col,
            prefix="",
            prefix_sep="",
            drop_first=False
        )
    
        dummies.columns = [f"{col}[{c}]" for c in dummies.columns]
        if impute_mode == "none":
            dummies.loc[df_proc[col].isna(), :] = np.nan
        dummies = dummies.astype(float)
    
        num_data_base = pd.concat([num_data_base, dummies], axis=1)
    
    # -------------------------
    # ④ 列順を揃える
    # -------------------------
    train_cols = explanatory_vars
    
    num_data = pd.DataFrame()
    for col in train_cols:
        if col in num_data_base.columns:
            num_data[col] = num_data_base[col]
        else:
            num_data[col] = 0.0
    
    df_encoded = num_data.copy()
    
    df_result = df_new.copy()

    # 学習データ（再学習用）
    df_num_train = pd.DataFrame(stored_num_data["data"], columns=stored_num_data["columns"])
    df_class_train = pd.DataFrame(stored_class_data["data"], columns=stored_class_data["columns"])
    df_train = pd.concat([df_num_train, df_class_train], axis=1)

    # --- 回帰予測 ---
    for i, (target, alpha, l1_ratio) in enumerate(zip(target_variables, alpha_values, l1_ratio_values)):
        dropped = df_train.dropna(subset=[target])
        X_train = dropped[explanatory_vars].copy()
        y_train = dropped[target].values
        
        preprocessor = NumericPreprocessor(impute_mode, scale=True)
        if regression_method == "huber":
            t = alpha / (1 - alpha)
            pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="huber", fit_params={"t": t})
        elif regression_method == "quantile":
            pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="quantile", fit_params={"q": alpha})
        else:
            model = ElasticNet(alpha=(10 ** alpha), l1_ratio=l1_ratio, random_state=0)
            pipeline = ModelPipeline(preprocessor, model)
            
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(df_encoded)
        
        df_result[target] = pred

    # --- 分類予測 ---
    for j, (c_target, alpha, l1_ratio) in enumerate(zip(class_variables, alpha_values[len(target_variables):], l1_ratio_values[len(target_variables):])):
        idx = len(target_variables) + j
        dropped = df_train.dropna(subset=[c_target])
        y_train = dropped[c_target].map({"O": 1, "X": 0}).astype(int).values
        X_train = dropped[explanatory_vars].copy()
        
        preprocessor = NumericPreprocessor(impute_mode, scale=True)
        if regression_method in ["quantile", "huber"]:
            pipeline = ModelPipeline(preprocessor, estimator=None, is_statsmodels=True, statsmodels_type="firth")
            pipeline.fit(X_train, y_train)
            prob = pipeline.predict_proba(df_encoded)
        else:
            model = LogisticRegression(penalty="elasticnet", solver="saga", C=1/(10 ** alpha), l1_ratio=l1_ratio, random_state=0)
            pipeline = ModelPipeline(preprocessor, model)
            pipeline.fit(X_train, y_train)
            prob = pipeline.predict_proba(df_encoded)[:, 1]
            
        threshold = threshold_values[idx]
        df_result[c_target] = np.where(prob >= threshold, "O", "X")
        df_result[f"{c_target}_proba"] = prob

    # -------------------------
    # ⑦ CSV 出力
    # -------------------------
    try:
        csv_bytes = df_result.to_csv(index=False, encoding="cp932").encode("cp932")
    except UnicodeEncodeError:
        csv_bytes = df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    b64 = base64.b64encode(csv_bytes).decode("ascii")
    base_name, _ = os.path.splitext(filename)
    new_filename = f"{base_name}_{regression_method}.csv"
    
    return (
        "予測完了",
        {"color": "green"},
        dict(content=b64, filename=new_filename, base64=True)
    )
    
def open_browser():
        webbrowser.open_new('http://127.0.0.1:8050/')
    
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=8050, debug=False)
