# %% [markdown]
# ## numexpr

# %%
from pandas.core.computation.check import NUMEXPR_INSTALLED

# %% [markdown]
# ## imports

# %%
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# %% [markdown]
# ## Anova Boxplot - working

# %%
df = px.data.gapminder() # use gapminder dataset for real-world data
df_2007 = df[df['year'] == 2007]

figsize = (800, 600)

fig = px.box(
    df_2007,
    x="continent",
    y="gdpPercap",
    points="all",
    title="ANOVA Box Plot: GDP per Capita by Continent (2007)"
)

groups = [df_2007[df_2007['continent'] == c]['gdpPercap'] for c in df_2007['continent'].unique()]
f_stat, p_val = stats.f_oneway(*groups)

fig.add_annotation(
    text=f"ANOVA F={f_stat:.2f}, p={p_val:.4f}",
    x=2,
    y=max(df_2007['gdpPercap']),
    showarrow=False,
    font=dict(size=12, color="red")
)

fig.update_layout(
    xaxis_title="Continent",
    yaxis_title="GDP per Capita (USD)",
    plot_bgcolor="white"
)

fig.show()

# %% [markdown]
# ## beeswarm plot - working

# %% [markdown]
# ### CREATE country data as data

# %%
data = {
    "Country": ["India", "United States", "China", "Nigeria", "Brazil"],
    "GDP_per_capita_USD": [2389, 76741, 12741, 2229, 9367],
    "Income_group": [
        "Lower-middle income",
        "High income",
        "Upper-middle income",
        "Lower-middle income",
        "Upper-middle income"
    ]
}

# %%
import plotly.express as px

fig = px.scatter(
    data,
    x="GDP_per_capita_USD",
    y=[0, 0, 0, 0, 0],  # simple flat y-axis (no jitter)
    color="Income_group",
    text="Country",
    labels={"GDP_per_capita_USD": "GDP per Capita (US$)", "y": ""},
    title="Beeswarm-style Plot of GDP per Capita by Income Group"
)

fig.update_traces(
    marker=dict(size=12, line=dict(width=0.5, color="DarkSlateGrey")),
    textposition="top center"
)

fig.update_layout(
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    xaxis_title="GDP per Capita (US$)",
    legend_title="World Bank Income Group",
    plot_bgcolor="white"
)

fig.show()

# %% [markdown]
# ## Density Heatmap vs scatterplot - working

# %% [markdown]
# ### CREATE Real sample data (World Bank 2023 values, approximate) as df

# %%

data = {
    "Country": [
        "India", "United States", "China", "Nigeria", "Brazil",
        "Japan", "Germany", "Ethiopia", "South Africa", "Italy"
    ],
    "GDP_per_capita": [
        2413, 76399, 12720, 2229, 9671,
        33950, 55400, 1271, 6776, 39500
    ],
    "Life_expectancy": [
        67.5, 76.4, 78.2, 54.0, 75.1,
        84.6, 81.0, 66.0, 64.0, 82.5
    ]
}

df = pd.DataFrame(data)

# %%
import pandas as pd
import plotly.express as px
import plotly.subplots as sp

# Density heatmap
fig_heatmap = px.density_heatmap(
    df,
    x="GDP_per_capita",
    y="Life_expectancy",
    nbinsx=20,
    nbinsy=20,
    color_continuous_scale="Viridis",
    title="Density Heatmap: GDP vs Life Expectancy"
)

# Scatter plot
fig_scatter = px.scatter(
    df,
    x="GDP_per_capita",
    y="Life_expectancy",
    text="Country",
    color="Country",
    opacity=0.8,
    title="Scatter Plot: GDP vs Life Expectancy"
)

# Combine into subplots
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Density Heatmap", "Scatter Plot"))

for trace in fig_heatmap.data:
    fig.add_trace(trace, row=1, col=1)

for trace in fig_scatter.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(
    xaxis_title="GDP per Capita (USD)",
    yaxis_title="Life Expectancy (Years)",
    plot_bgcolor="white",
    title_text="Comparison: Density Heatmap vs Scatter Plot (World Bank 2023)"
)

fig.show()

# %% [markdown]
# ## facet histogram - working

# %% [markdown]
# ### CREATE countries dataset as df

# %%
data = {
    "Country": ["India", "United States", "China", "Nigeria", "Brazil"],
    "GDP_per_capita_USD": [2389, 76741, 12741, 2229, 9367],  # 2022 values
    "Income_group": ["Lower-middle income", "High income", "Upper-middle income", "Lower-middle income", "Upper-middle income"]
}

df = pd.DataFrame(data)

# %%
import pandas as pd
import plotly.express as px

fig = px.histogram(
    df,
    x="GDP_per_capita_USD",
    color="Income_group",
    facet_col="Income_group",
    title="Facet Histogram: GDP per Capita by Income Group (2022 World Bank values)"
)

fig.update_layout(
    xaxis_title="GDP per Capita (US$)",
    yaxis_title="Count",
    plot_bgcolor="white"
)

fig.show()
plt.close("all")

# %% [markdown]
# ## organizational chart - working

# %%
import plotly.graph_objects as go

labels2 = ["CEO", "CTO", "CFO", "COO", 
          "Engineering Manager", "Finance Manager", "Operations Manager",
          "Developer A", "Developer B", "Accountant", "Ops Staff"]

parents2 = ["", "CEO", "CEO", "CEO", 
           "CTO", "CFO", "COO", 
           "Engineering Manager", "Engineering Manager", "Finance Manager", "Operations Manager"]
labels = []
parents = []
fig = go.Figure(go.Sunburst(
    labels=labels2,
    parents=parents2,
    marker=dict(colors=["#636EFA","#EF553B","#00CC96","#AB63FA",
                        "#FFA15A","#19D3F3","#FF6692",
                        "#B6E880","#FF97FF","#FECB52","#9D9D9D"]),
    branchvalues="total"
))

fig.update_layout(
    title="Organizational Chart (Company Hierarchy)",
    margin=dict(t=40, l=0, r=0, b=0)
)

fig.show()

# %% [markdown]
# ## taxonomy tree - working

# %%
import plotly.graph_objects as go

labels = [
    "Life", 
    "Domain: Eukarya", "Domain: Bacteria", "Domain: Archaea",
    "Kingdom: Animalia", "Kingdom: Plantae", "Kingdom: Fungi",
    "Phylum: Chordata", "Phylum: Arthropoda",
    "Class: Mammalia", "Class: Insecta",
    "Species: Human", "Species: Tiger", "Species: Butterfly", "Species: Ant"
]

parents = [
    "", 
    "Life", "Life", "Life",
    "Domain: Eukarya", "Domain: Eukarya", "Domain: Eukarya",
    "Kingdom: Animalia", "Kingdom: Animalia",
    "Phylum: Chordata", "Phylum: Arthropoda",
    "Class: Mammalia", "Class: Mammalia", "Class: Insecta", "Class: Insecta"
]

fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    branchvalues="total"
))

fig.update_layout(
    title="Taxonomy Tree (Classification Hierarchy)",
    margin=dict(t=40, l=0, r=0, b=0)
)

fig.show()

# %% [markdown]
# ## sunburst chart - working

# %%
import plotly.graph_objects as go

labels = [
    "India",
    "Rural", "Urban",
    "Maharashtra", "Bihar", "Delhi",
    "District A", "District B", "District C",
    "SC", "ST", "OBC", "General",
    "Low Income", "Middle Income", "High Income"
]

parents = [
    "",                # India is root
    "India", "India",  # Rural, Urban
    "Rural", "Rural", "Urban",   # States
    "Maharashtra", "Bihar", "Delhi",  # Districts
    "District A", "District A", "District B", "District C",  # Castes
    "SC", "ST", "OBC", "General"  # Income brackets
]

values = [
    200,  # India (root total)
    120,  # Rural
    80,   # Urban
    60, 40, 30,   # States
    25, 20, 15,   # Districts
    10, 8, 12, 5, # Castes
    6, 4, 7, 3    # Income brackets
]

fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total"
))

fig.update_layout(
    title="Sunburst Chart with Multiple Rings (Deep Hierarchy)",
    margin=dict(t=40, l=0, r=0, b=0)
)

fig.show()


# %% [markdown]
# ## icicle chart - working

# %%
import plotly.graph_objects as go
import plotly.io as pio

# Force a VS Code/Jupyter-friendly renderer
pio.renderers.default = "plotly_mimetype"

labels = [
    "India",
    "Rural", "Urban",
    "Maharashtra", "Bihar", "Delhi",
    "District A", "District B", "District C",
    "SC", "ST", "OBC", "General",
    "Low Income", "Middle Income", "High Income"
]

parents = [
    "",                # India is root
    "India", "India",  # Rural, Urban
    "Rural", "Rural", "Urban",   # States
    "Maharashtra", "Bihar", "Delhi",  # Districts
    "District A", "District A", "District B", "District C",  # Castes
    "SC", "ST", "OBC", "General"  # Income brackets
]

values = [
    200,  # India (root total)
    120,  # Rural
    80,   # Urban
    60, 40, 30,   # States
    25, 20, 15,   # Districts
    10, 8, 12, 5, # Castes
    6, 4, 7, 3    # Income brackets
]

fig = go.Figure(go.Icicle(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total"
))

fig.update_layout(
    title="Icicle Chart (Vertical Hierarchy: SECC-style Household Classification)",
    margin=dict(t=40, l=0, r=0, b=0)
)

fig.show(renderer="plotly_mimetype")

# %% [markdown]
# ## bipartite graph - working

# %%
import networkx as nx
import plotly.graph_objects as go

NGOs = ["NGO A", "NGO B", "NGO C"]
Groups = ["Women", "Children", "Farmers"]

edges = [
    ("NGO A", "Women"),
    ("NGO A", "Children"),
    ("NGO B", "Children"),
    ("NGO B", "Farmers"),
    ("NGO C", "Women"),
    ("NGO C", "Farmers")
]

B = nx.Graph()
B.add_nodes_from(NGOs, bipartite=0)
B.add_nodes_from(Groups, bipartite=1)
B.add_edges_from(edges)

pos = {}
pos.update((node, (0, i)) for i, node in enumerate(NGOs))   # NGOs on left
pos.update((node, (1, i)) for i, node in enumerate(Groups)) # Groups on right

edge_x = []
edge_y = []
for u, v in B.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y,
                        line=dict(width=1, color='gray'),
                        hoverinfo='none',
                        mode='lines')

node_x = []
node_y = []
node_text = []
node_text_positions = []
for node in B.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_text_positions.append("middle right" if x == 0 else "middle left")

node_trace = go.Scatter(x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition=node_text_positions,
                        cliponaxis=False,
                        marker=dict(size=20, color='skyblue'),
                        hoverinfo='text')

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title="Bipartite Graph: NGOs vs Beneficiary Groups",
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.25, 1.25]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.25, 2.25])
)

fig.show()

# %% [markdown]
# ### CREATE SYNTRA dataset as df

# %%
#Syntra dataset
data = [
    {"item_id": 1, "product_name": "Syntra Slim Jeans",     "category": "Jeans",      "price": 1999, "sales": 250, "inventory": 120, "region": "India"},
    {"item_id": 2, "product_name": "Syntra Linen Shirt",    "category": "Shirts",     "price": 1499, "sales": 320, "inventory": 150, "region": "India"},
    {"item_id": 3, "product_name": "Syntra Hoodie",         "category": "Outerwear",  "price": 2499, "sales": 210, "inventory": 80,  "region": "India"},
    {"item_id": 4, "product_name": "Syntra Chino Shorts",   "category": "Shorts",     "price": 1299, "sales": 180, "inventory": 70,  "region": "South Asia"},
    {"item_id": 5, "product_name": "Syntra Graphic Tee",    "category": "T-Shirts",   "price": 999,  "sales": 500, "inventory": 200, "region": "South Asia"},
    {"item_id": 6, "product_name": "Syntra Cargo Pants",    "category": "Pants",      "price": 1799, "sales": 230, "inventory": 100, "region": "India"},
    {"item_id": 7, "product_name": "Syntra Denim Jacket",   "category": "Outerwear",  "price": 2999, "sales": 150, "inventory": 60,  "region": "India"},
    {"item_id": 8, "product_name": "Syntra Polo Shirt",     "category": "Shirts",     "price": 1599, "sales": 270, "inventory": 90,  "region": "South Asia"},
    {"item_id": 9, "product_name": "Syntra Sweatpants",     "category": "Pants",      "price": 1399, "sales": 300, "inventory": 130, "region": "India"},
    {"item_id":10, "product_name": "Syntra Summer Dress",   "category": "Dresses",    "price": 2199, "sales": 190, "inventory": 70,  "region": "South Asia"},
]

df = pd.DataFrame(data)

# %% [markdown]
# ## dendogram - working

# %%
import pandas as pd
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage

# Select features for clustering
X = df[['price', 'sales', 'inventory']].values   # gives NumPy array internally
labels = df['product_name'].tolist()

# Perform hierarchical clustering 
Z = linkage(X, method='ward')

# Create dendrogram 
fig = ff.create_dendrogram(X, labels=labels, linkagefun=lambda x: linkage(x, 'ward'))
fig.update_layout(width=1000, height=600, title="Syntra Products - Dendrogram (No NumPy Import)")

fig.show()

# %%
import plotly.express as px

# %% [markdown]
# ### READ-IN SID dataset as df

# %%
df = pd.read_csv("SID_renamed.csv")

# %% [markdown]
# ## faceted scatter - working

# %%
# Load CSV file
# CSV must have: product_name, category, price, sales, inventory, region
# df = pd.read_csv("dcd25_renamed.csv")

# Faceted scatter plot
fig = px.scatter(
    df,
    x="region4",
    y="region9",
    color="stabbr",
    facet_col="region4",
    hover_name="state_name",
    size="statefip",
    title="SID dataset"
)

fig.update_layout(width=1200, height=600)
fig.show()

# %% [markdown]
# ## mosaic like - working

# %%
import pandas as pd
import plotly.express as px

# Mosaic-like plot using Treemap
fig = px.treemap(
    df,
    path=["region4", "region9"],  # hierarchy: Category → Region
    values="year",               # area proportional to sales
    color="region4",
    # hover_data=["state_name", "statefip", "effort"],
    title="SID Dataset - Mosaic Plot (Category × Region)"
)

fig.update_layout(width=900, height=600)
fig.show()

# %%
import pandas as pd
import plotly.graph_objects as go
import circlify

# %% [markdown]
# ### CREATE syntra dataset as df

# %%
#Syntra dataset
data = [
    {"item_id": 1, "product_name": "Syntra Slim Jeans",     "category": "Jeans",      "price": 1999, "sales": 250, "inventory": 120, "region": "India"},
    {"item_id": 2, "product_name": "Syntra Linen Shirt",    "category": "Shirts",     "price": 1499, "sales": 320, "inventory": 150, "region": "India"},
    {"item_id": 3, "product_name": "Syntra Hoodie",         "category": "Outerwear",  "price": 2499, "sales": 210, "inventory": 80,  "region": "India"},
    {"item_id": 4, "product_name": "Syntra Chino Shorts",   "category": "Shorts",     "price": 1299, "sales": 180, "inventory": 70,  "region": "South Asia"},
    {"item_id": 5, "product_name": "Syntra Graphic Tee",    "category": "T-Shirts",   "price": 999,  "sales": 500, "inventory": 200, "region": "South Asia"},
    {"item_id": 6, "product_name": "Syntra Cargo Pants",    "category": "Pants",      "price": 1799, "sales": 230, "inventory": 100, "region": "India"},
    {"item_id": 7, "product_name": "Syntra Denim Jacket",   "category": "Outerwear",  "price": 2999, "sales": 150, "inventory": 60,  "region": "India"},
    {"item_id": 8, "product_name": "Syntra Polo Shirt",     "category": "Shirts",     "price": 1599, "sales": 270, "inventory": 90,  "region": "South Asia"},
    {"item_id": 9, "product_name": "Syntra Sweatpants",     "category": "Pants",      "price": 1399, "sales": 300, "inventory": 130, "region": "India"},
    {"item_id":10, "product_name": "Syntra Summer Dress",   "category": "Dresses",    "price": 2199, "sales": 190, "inventory": 70,  "region": "South Asia"},
]

df = pd.DataFrame(data)

# %% [markdown]
# ## circlify - working

# %%
# Circle packing layout (by sales)
circles = circlify.circlify(
    df['sales'].tolist(),
    show_enclosure=False,
    target_enclosure=circlify.Circle(x=0, y=0, r=1)
)

# Create figure with circles
fig = go.Figure()

for i, circle in enumerate(circles):
    if i >= len(df):
        continue
    x, y, r = circle.x, circle.y, circle.r
    row = df.iloc[i]
    fig.add_shape(
        type="circle",
        x0=x-r, y0=y-r,
        x1=x+r, y1=y+r,
        line=dict(color="black", width=1),
        fillcolor="lightblue"
    )
    # Number label inside circle
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        text=[str(i+1)],
        mode="text",
        hovertext=f"{row['product_name']}<br>Sales: {row['sales']}<br>Category: {row['category']}",
        hoverinfo="text",
        textfont=dict(size=14, color="black", family="Arial Bold")
    ))

fig.update_layout(
    width=700, height=700,
    title="Syntra Products - Circular Packing (Numbered Bubbles by Sales)",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    showlegend=False
)

fig.show()

# %% [markdown]
# ## quantile quantile - working

# %%
# Imports
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

# Load dataset from CSV
# Your CSV must have at least: product_name, price
# df = pd.read_csv("syntra_products.csv")

# Pick a variable for QQ-plot (e.g., price column)
values = df['price']

# Generate QQ-plot data
osm, osr = stats.probplot(values, dist="norm")[:2]

# Create figure
fig = go.Figure()

# Scatter points (Observed vs Theoretical quantiles)
fig.add_trace(go.Scatter(
    x=osm[0], y=osm[1],
    mode="markers+text",
    text=df['product_name'],
    textposition="top center",
    name="Products"
))

# Reference line (perfect normal fit)
slope, intercept, r = osr
x_line = [min(osm[0]), max(osm[0])]
y_line = [slope*x + intercept for x in x_line]
fig.add_trace(go.Scatter(
    x=x_line, y=y_line,
    mode="lines",
    name="Normal line",
    line=dict(color="red", dash="dash")
))

# Layout
fig.update_layout(
    title="QQ-Plot of Syntra Product Prices vs Normal Distribution",
    xaxis_title="Theoretical Quantiles (Normal Dist.)",
    yaxis_title="Observed Product Prices",
    width=800,
    height=600
)

fig.show()

# %% [markdown]
# ## spie chart of categories - working

# %%
import pandas as pd
import plotly.graph_objects as go

# Load your CSV file (replace with your file name/path)
# df = pd.read_csv("your_file.csv")

agg = df.groupby("category").agg({
    "sales": "sum",
    "price": "mean"
}).reset_index()

agg["angle"] = agg["sales"] / agg["sales"].sum() * 360
agg["radius"] = agg["price"] / agg["price"].max() * 100

fig = go.Figure()

fig.add_trace(go.Barpolar(
    r=agg["radius"],
    theta=agg["angle"].cumsum() - agg["angle"]/2,
    width=agg["angle"],
    text=agg["category"] + "<br>Sales=" + agg["sales"].astype(str) + "<br>Avg Price=" + agg["price"].astype(str),
    hoverinfo="text",
    marker=dict(line=dict(color="black", width=1))
))

fig.update_layout(
    title="Spie Chart of Categories (Sales vs Price)",
    polar=dict(
        radialaxis=dict(showticklabels=False, ticks=''),
        angularaxis=dict(showticklabels=False, ticks='')
    ),
    showlegend=False,
    width=800,
    height=600
)

fig.show()

print("Category Breakdown:")
print(agg[["category", "sales", "price"]])

# %% [markdown]
# ## aggregate by category - working

# %%

# Aggregate by category → min, max, mean of price & sales
agg = df.groupby("category").agg({
    "price": ["min", "max", "mean"],
    "sales": ["min", "max", "mean"]
}).reset_index()

# Flatten column names
agg.columns = ["category", "price_min", "price_max", "price_mean", "sales_min", "sales_max", "sales_mean"]

# Plot Tufte Min–Max for Price 
fig = go.Figure()

for i, row in agg.iterrows():
    # Thin line (min to max)
    fig.add_trace(go.Scatter(
        x=[row["price_min"], row["price_max"]],
        y=[row["category"], row["category"]],
        mode="lines",
        line=dict(color="black", width=1),
        showlegend=False
    ))

    # Dot (mean)
    fig.add_trace(go.Scatter(
        x=[row["price_mean"]],
        y=[row["category"]],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Mean Price" if i == 0 else None
    ))

fig.update_layout(
    title="Tufte Min–Max Plot of Categories (Price)",
    xaxis_title="Price",
    yaxis_title="Category",
    template="simple_white",
    width=800,
    height=500
)

fig.show()

# Side table for reference
print(agg[["category", "price_min", "price_max", "price_mean"]])

# %% [markdown]
# ## ternary plot

# %% [markdown]
# ### CREATE ternaryplot_df

# %%
# Example dataset for Ternary Plot 
ternaryplot_df = pd.DataFrame({
    "A": [0.1, 0.3, 0.6, 0.2, 0.4],
    "B": [0.6, 0.3, 0.2, 0.7, 0.4],
    "C": [0.3, 0.4, 0.2, 0.1, 0.2],
    "Label": ["P1", "P2", "P3", "P4", "P5"]
})

# %% [markdown]
# ## sankey - working

# %%
# Extracting 
sources = df['product_name']
targets = df['category']
suppliers = df['region']

# Creating unique labels for all nodes in the Sankey diagram
all_nodes = list(pd.concat([sources, targets, suppliers]).unique())
node_map = {name: idx for idx, name in enumerate(all_nodes)}

# flow values: Channel → Category → Supplier
flows = (
    df.groupby(['product_name', 'category', 'region'])
      .size()
      .reset_index(name='Value')  # 'Value' represents the count of each flow
)

# Mapping each node to its index
flows['Source'] = flows['product_name'].map(node_map)
flows['Middle'] = flows['category'].map(node_map)
flows['Target'] = flows['region'].map(node_map)

# Creating links for the Sankey diagram:
links = pd.DataFrame({
    'source': list(flows['Source']) + list(flows['Middle']),
    'target': list(flows['Middle']) + list(flows['Target']),
    'value': list(flows['Value']) + list(flows['Value'])
})

# Building the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=25,  # Space between nodes
        thickness=20,  # Thickness of each node
        line=dict(color="black", width=0.8),  # Border around nodes
        label=all_nodes,  # Node labels
        color="lightblue"  # Node color
    ),
    link=dict(
        source=links['source'],  # Source node indices
        target=links['target'],  # Target node indices
        value=links['value'],    # Flow values
        color="rgba(50,150,250,0.4)"  # Link color with transparency
    )
)])

# Final layout settings
fig.update_layout(title_text="Syntra Clothing - Sankey with Loops", font_size=12)
fig.show()

# %% [markdown]
# ## tornado chart - working

# %%
import pandas as pd
import plotly.graph_objects as go

# Validate required columns
required_cols = {"category", "region", "sales"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

# Aggregate sales separately for category and region
category_totals = df.groupby("category", dropna=False)["sales"].sum()
region_totals = df.groupby("region", dropna=False)["sales"].sum()

# Use a unified y-axis so both groups can be mirrored in one tornado chart
labels = sorted(set(category_totals.index.astype(str)).union(region_totals.index.astype(str)))
category_lookup = category_totals.rename_axis("label").rename("sales").reset_index()
region_lookup = region_totals.rename_axis("label").rename("sales").reset_index()

category_map = dict(zip(category_lookup["label"].astype(str), category_lookup["sales"]))
region_map = dict(zip(region_lookup["label"].astype(str), region_lookup["sales"]))

category_sales = [float(category_map.get(lbl, 0)) for lbl in labels]
region_sales = [float(region_map.get(lbl, 0)) for lbl in labels]
category_sales_negative = [-x for x in category_sales]

max_val = max(category_sales + region_sales) if (category_sales or region_sales) else 0

fig = go.Figure()

fig.add_trace(go.Bar(
    y=labels,
    x=category_sales_negative,
    orientation="h",
    name="Category Sales",
    marker=dict(color="#f28e2b")
))

fig.add_trace(go.Bar(
    y=labels,
    x=region_sales,
    orientation="h",
    name="Region Sales",
    marker=dict(color="#4e79a7")
))

fig.update_layout(
    title="Tornado Chart: Category vs Region Sales",
    barmode="relative",
    bargap=0.15,
    xaxis=dict(
        tickvals=[-max_val, 0, max_val],
        ticktext=[str(int(max_val)), "0", str(int(max_val))],
        title="Sales"
    ),
    yaxis=dict(title="Category/Region"),
    template="simple_white",
    width=900,
    height=600
)

fig.show()

# %% [markdown]
# ### CREATE df_locations from df

# %%
import pandas as pd

# Build a dataframe that matches the location cell requirements: Store_Location and Sales
if {"Store_Location", "Sales"}.issubset(df.columns):
    df_locations = df[["Store_Location", "Sales"]].copy()
elif {"region", "sales"}.issubset(df.columns):
    # Convert Syntra-style schema to location schema
    region_to_city = {
        "India": "Mumbai",
        "South Asia": "Delhi"
    }
    df_locations = df[["region", "sales"]].copy()
    df_locations["Store_Location"] = df_locations["region"].map(region_to_city).fillna("Mumbai")
    df_locations["Sales"] = pd.to_numeric(df_locations["sales"], errors="coerce").fillna(0)
    df_locations = df_locations[["Store_Location", "Sales"]]
elif {"Region", "Sales"}.issubset(df.columns):
    # Convert title-case schema to expected names
    df_locations = df[["Region", "Sales"]].copy()
    df_locations = df_locations.rename(columns={"Region": "Store_Location"})
else:
    # Safe fallback sample so the location cell can still run
    df_locations = pd.DataFrame({
        "Store_Location": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"],
        "Sales": [220, 180, 140, 120, 100]
    })

# Ensure required cities are present in the dataframe
required_cities = ["Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
existing_cities = set(df_locations["Store_Location"].astype(str))
missing_cities = [city for city in required_cities if city not in existing_cities]

if missing_cities:
    default_sales = float(df_locations["Sales"].median()) if not df_locations.empty else 100.0
    add_rows = pd.DataFrame({
        "Store_Location": missing_cities,
        "Sales": [default_sales] * len(missing_cities)
    })
    df_locations = pd.concat([df_locations, add_rows], ignore_index=True)

df_locations

# %% [markdown]
# ## location plot - working

# %%
import plotly.express as px

# Example mapping for Indian cities (add more if needed)
city_coords = {
    "Mumbai": [19.0760, 72.8777],
    "Delhi": [28.7041, 77.1025],
    "Bangalore": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639],
    "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567],
    "Ahmedabad": [23.0225, 72.5714]
}

# Add Latitude & Longitude columns to df_locations
df_locations = df_locations.copy()
df_locations["Latitude"] = df_locations["Store_Location"].map(lambda x: city_coords.get(x, [0, 0])[0])
df_locations["Longitude"] = df_locations["Store_Location"].map(lambda x: city_coords.get(x, [0, 0])[1])

fig = px.density_mapbox(
    df_locations,
    lat="Latitude",
    lon="Longitude",
    z="Sales",
    radius=30,
    center=dict(lat=20.5937, lon=78.9629),
    zoom=4,
    mapbox_style="carto-positron"
)

fig.update_layout(
    title="Syntra Clothing - Hexbin Sales Density Map"
)

fig.show()

# %% [markdown]
# ## bullet chart - working

# %%
import plotly.graph_objects as pogo

# Expected columns: Category, Sales, Target_Sales

fig = pogo.Figure()

# Target (background bars)
fig.add_trace(pogo.Bar(
    x=df['sales']+100,  # Add some padding to show the target bar behind
    y=df['category'],
    orientation='h',
    marker=dict(color='lightgray'),
    name='Target'
))

# Actual (foreground bars)
fig.add_trace(pogo.Bar(
    x=df['sales'],
    y=df['category'],
    orientation='h',
    marker=dict(color='blue'),
    name='Actual'
))

fig.update_layout(
    title="Syntra Clothing - Bullet Chart (Sales vs Target)",
    barmode='overlay',
    bargap=0.3,
    xaxis_title="Sales"
)

fig.show()

# %% [markdown]
# ## ternary plot - working

# %%
import plotly.express as px
import pandas as pd

# Example dataset for Ternary Plot 
ternary_df = pd.DataFrame({
    "A": [0.1, 0.3, 0.6, 0.2, 0.4],
    "B": [0.6, 0.3, 0.2, 0.7, 0.4],
    "C": [0.3, 0.4, 0.2, 0.1, 0.2],
    "Label": ["P1", "P2", "P3", "P4", "P5"]
})

# Ternary scatter plot
fig = px.scatter_ternary(
    ternary_df,
    a="A", b="B", c="C",
    hover_name="Label",
    color="Label",
    size=[10,20,15,25,18],
    title="Ternary Plot (A-B-C Proportions)"
)

fig.show()

# %% [markdown]
# ## radar chart - working

# %%
import plotly.express as px
radar_df = df.pivot_table(
    index='category', 
    columns='region', 
    values='sales', 
    aggfunc='sum'
).reset_index()

# Melt dataframe for px.line_polar
radar_melted = radar_df.melt(id_vars='category', var_name='region', value_name='sales')

# Radar chart
fig = px.line_polar(
    radar_melted,
    r='sales',
    theta='region',
    color='category',
    line_close=True,
    title='Syntra Clothing - Radar Chart (Category vs Region Sales)'
)

fig.show()

# %% [markdown]
# ## polar area - working

# %%
import plotly.express as px

# Using uploaded df
# Columns expected: Region, Category, Sales

fig = px.bar_polar(
    df,
    r="sales",
    theta="region",   # Each region around the circle
    color="category",
    template="plotly_dark",
    title="Syntra Clothing - Polar Area / Wind Rose Chart",
    color_discrete_sequence=px.colors.sequential.Plasma_r
)

fig.show()

# %% [markdown]
# ## chord diagram - working

# %%
import plotly.graph_objects as go
import pandas as pd
import numpy as np

categories = df['category'].unique().tolist()

np.random.seed(0)
matrix = np.random.randint(0, 100, size=(len(categories), len(categories)))

sources = []
targets = []
values = []

for i in range(len(categories)):
    for j in range(len(categories)):
        if matrix[i, j] > 0:
            sources.append(i)
            targets.append(j)
            values.append(matrix[i, j])

fig = go.Figure(data=[go.Sankey(
    node=dict(
        label=categories,
        pad=15,
        thickness=20,
        color="lightgreen"
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color="rgba(50,150,250,0.5)"
    )
)])

fig.update_layout(
    title_text="Syntra Clothing - Chord Diagram (Approximation)",
    font_size=12
)

fig.show()

# %% [markdown]
# ## chart1  - working

# %%
import plotly.graph_objects as go

# Create the table with four columns
fig = go.Figure(data=[go.Table(
    columnwidth=[50, 200, 80, 100],
    header=dict(
        values=['<b>Tool</b>', '<b>Brief Explanation</b>', '<b>Level of Difficulty</b>', '<b>Website</b>'],
        fill_color='hotpink',
        align='left',
        font=dict(color='white', size=12, family='Poppins')
    ),
    cells=dict(
        values=[
            ['NotebookLM', 'Napkin AI', 'Firebase'],
            ['AI-powered research and note-taking assistant that helps organize and analyze documents', 
             'AI tool that transforms text into visual diagrams and infographics automatically', 
             'Google cloud platform providing backend services like databases, authentication, and hosting'],
            ['Beginner', 'Beginner', 'Intermediate'],
            ['<a href="https://notebooklm.google.com">notebooklm.google.com</a>',
             '<a href="https://napkin.ai">napkin.ai</a>',
             '<a href="https://firebase.google.com">firebase.google.com</a>']
        ],
        fill_color=[['#f0f0f0', 'lightgray', '#f0f0f0']],  # Lighter gray alternating with medium gray
        align='left',
        font=dict(size=11, family='Poppins')
    ))
])

fig.update_layout(width=800, 
                  height=250,
                  margin=dict(l=10, r=10, t=60, b=10))
fig.show()

# %% [markdown]
# ## chart2 - working

# %%

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    columnwidth=[50, 200, 80, 100],
    header=dict(
        values=['<b>Tool</b>', '<b>Brief Explanation</b>', '<b>Level of Difficulty</b>', '<b>Website</b>'],
        fill_color='royalblue',
        align=['left', 'center', 'center', 'left'],
        font=dict(color='white', size=14, family='Poppins'),
        line=dict(color='darkblue', width=2),
        height=40
    ),
    cells=dict(
        values=[
            ['NotebookLM', 'Napkin AI', 'Firebase'],
            ['AI-powered research and note-taking assistant', 
             'AI tool that transforms text into visual diagrams', 
             'Google cloud platform for backend services'],
            ['Beginner', 'Beginner', 'Intermediate'],
            ['<a href="https://notebooklm.google.com">notebooklm.google.com</a>',
             '<a href="https://napkin.ai">napkin.ai</a>',
             '<a href="https://firebase.google.com">firebase.google.com</a>']
        ],
        # Customize colors per column and per row
        fill_color=[
            ['#f0f0f0', 'lightgray', '#f0f0f0'],           # Tool column - alternating gray
            ['#f0f0f0', 'lightgray', '#f0f0f0'],           # Brief Explanation column - alternating gray
            ['#90EE90', '#90EE90', '#FFB347'],           # Difficulty: Green for Beginner, Orange for Intermediate
            ['#f0f0f0', 'lightgray', '#f0f0f0']            # Website column - alternating gray
        ],
        align=['left', 'left', 'center', 'left'],
        font=dict(size=11, family='Poppins', color='#333'),
        line=dict(color='gray', width=1),
        height=30
    ))
])

fig.update_layout(
    width=800, 
    height=250,
    title='AI Tools Comparison',
    title_font_size=20,
    margin=dict(l=10, r=10, t=60, b=10)
)

fig.show()

# %% [markdown]
# ## dendogram horizontal - working

# %%
import plotly.figure_factory as ff
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Estimated income and employment values; not official figures
districts_data = {
    'District': [
        'Salamanca', 'Chamartín', 'Retiro', 'Chamberí', 'Centro',
        'Moncloa-Aravaca', 'Tetuán', 'Arganzuela', 'Fuencarral-El Pardo', 'Hortaleza',
        'Ciudad Lineal', 'Latina', 'Carabanchel', 'Usera', 'Puente de Vallecas',
        'Moratalaz', 'San Blas-Canillejas', 'Barajas', 'Vicálvaro', 'Villa de Vallecas', 'Villaverde'
    ],
    # Average annual income in EUR
    'Average_Income_EUR': [
        52000, 48000, 42000, 40000, 35000,
        45000, 30000, 32000, 38000, 36000,
        28000, 26000, 24000, 22000, 23000,
        29000, 27000, 31000, 25000, 24000, 23000
    ],
    # Employment rate (%)
    'Employment_Rate': [
        94.5, 93.8, 92.5, 92.0, 88.0,
        93.0, 87.5, 89.0, 91.5, 91.0,
        88.5, 87.0, 85.5, 84.0, 85.0,
        88.8, 87.2, 90.0, 86.0, 85.5, 84.5
    ]
}

# Create DataFrame
df = pd.DataFrame(districts_data)

# Select only income and employment rate
X = df[['Average_Income_EUR', 'Employment_Rate']].values

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Create dendrogram
fig = ff.create_dendrogram(
    X_standardized,
    orientation='left',
    labels=df['District'].tolist(),
    colorscale=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C'],
    color_threshold=1.5  # Adjust this value to show more/fewer colors (try 0.5, 1.0, 1.5, 2.0)
)

fig.update_layout(
    width=900,
    height=600,
    title=dict(
        text='Madrid Districts Hierarchical Clustering (2025)<br><sub>Based on Average Income and Employment Rate</sub>',
        font=dict(size=20, family='Poppins', color='#2C3E50')
    ),
    xaxis=dict(
        title='Euclidean Distance (Standardized)',
        title_font=dict(size=14, family='Poppins'),
        tickfont=dict(size=11, family='Poppins'),
        gridcolor='#E0E0E0'
    ),
    yaxis=dict(
        tickfont=dict(size=12, family='Poppins')
    ),
    font=dict(family='Poppins'),
    plot_bgcolor='#FAFAFA',
    paper_bgcolor='white',
    margin=dict(l=180, r=50, t=120, b=80)
)

fig.show()

# %% [markdown]
# ## dendogram vertical - working

# %%
import plotly.figure_factory as ff
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Estimated income and employment values; not official figures
districts_data = {
    'District': [
        'Salamanca', 'Chamartín', 'Retiro', 'Chamberí', 'Centro',
        'Moncloa-Aravaca', 'Tetuán', 'Arganzuela', 'Fuencarral-El Pardo', 'Hortaleza',
        'Ciudad Lineal', 'Latina', 'Carabanchel', 'Usera', 'Puente de Vallecas',
        'Moratalaz', 'San Blas-Canillejas', 'Barajas', 'Vicálvaro', 'Villa de Vallecas', 'Villaverde'
    ],
    # Average annual income in EUR
    'Average_Income_EUR': [
        52000, 48000, 42000, 40000, 35000,
        45000, 30000, 32000, 38000, 36000,
        28000, 26000, 24000, 22000, 23000,
        29000, 27000, 31000, 25000, 24000, 23000
    ],
    # Employment rate (%)
    'Employment_Rate': [
        94.5, 93.8, 92.5, 92.0, 88.0,
        93.0, 87.5, 89.0, 91.5, 91.0,
        88.5, 87.0, 85.5, 84.0, 85.0,
        88.8, 87.2, 90.0, 86.0, 85.5, 84.5
    ]
}

# Create DataFrame
df = pd.DataFrame(districts_data)

# Select only income and employment rate
X = df[['Average_Income_EUR', 'Employment_Rate']].values

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Create dendrogram
# MODIFICATION 1: Changed orientation to 'bottom'
# MODIFICATION 2: Changed to pastel colors
fig = ff.create_dendrogram(
    X_standardized,
    orientation='bottom',  # CHANGED from 'left'
    labels=df['District'].tolist(),
    colorscale=['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#E0BBE4', '#FFD9B3'],  # PASTEL COLORS
    color_threshold=1.5
)

# MODIFICATION 3: Increase line width
for trace in fig.data:
    trace.line.width = 3

fig.update_layout(
    width=900,
    height=500,
    title=dict(
        text='Madrid Districts Hierarchical Clustering (2025)<br><sub>Based on Average Income and Employment Rate</sub>',
        font=dict(size=20, family='Poppins', color='#2C3E50')
    ),
    xaxis=dict(
        title='Euclidean Distance (Standardized)',
        title_font=dict(size=14, family='Poppins'),
        tickfont=dict(size=11, family='Poppins'),
        gridcolor='#E0E0E0',
        mirror=False  # MODIFICATION 4: No mirror axis
    ),
    yaxis=dict(
        tickfont=dict(size=12, family='Poppins'),
        mirror=False  # MODIFICATION 4: No mirror axis
    ),
    font=dict(family='Poppins'),
    plot_bgcolor='#F0F0F0',  # MODIFICATION 5: Light gray background
    paper_bgcolor='white',
    margin=dict(l=180, r=50, t=120, b=80)
)

fig.show()

# %% [markdown]
# ## icicle chart1 - working

# %%
import plotly.express as px
import pandas as pd

# Create the data
languages = ["German", "German", "German", "English", "English", "English", 
             "Italian", "Italian", "Italian"]
levels = ["Beginner", "Intermediate", "Advanced", "Beginner", "Intermediate", "Advanced",
          "Beginner", "Intermediate", "Advanced"]
students = [45, 38, 22, 62, 51, 33, 28, 24, 15]

# Create DataFrame
df = pd.DataFrame({
    'language': languages,
    'level': levels,
    'students': students
})

# Add root node
df['academy'] = 'Language Academy'

# Create icicle chart
fig = px.icicle(df, 
                path=['academy', 'language', 'level'], 
                values='students',
                color='students',
                color_continuous_scale='Blues')

# Update traces
fig.update_traces(
    root_color='lightgrey', 
    textinfo='label+value+percent parent',
)

fig.update_layout(
    title='Language Academy Student Distribution',
    margin=dict(t=50, l=25, r=25, b=25),
    width=900,
    height=500
)

fig.show()

# %% [markdown]
# ## icicle chart2 - working

# %%
import plotly.express as px
import pandas as pd

# Create the data
languages = ["German", "German", "German", "English", "English", "English", 
             "Italian", "Italian", "Italian"]
levels = ["Beginner", "Intermediate", "Advanced", "Beginner", "Intermediate", "Advanced",
          "Beginner", "Intermediate", "Advanced"]
students = [45, 38, 22, 62, 51, 33, 28, 24, 15]

# Create DataFrame
df = pd.DataFrame({
    'language': languages,
    'level': levels,
    'students': students
})

# Add root node
df['academy'] = 'Language Academy'

# Create icicle chart with discrete colors
fig = px.icicle(df, 
                path=['academy', 'language', 'level'], 
                values='students',
                color='language',  # Color by language for discrete mapping
                color_discrete_map={
                    'German': '#1f77b4',      # Blue
                    'English': '#ff7f0e',     # Orange
                    'Italian': '#2ca02c',     # Green
                    '(?)': 'lightgrey'        # Root node
                })

# Update traces with white borders
fig.update_traces(
    root_color='lightgrey', 
    textinfo='label+value+percent parent',
    marker=dict(
        line=dict(color='white', width=4)  # Bigger white borders
    )
)

fig.update_layout(
    title='Language Academy Student Distribution',
    margin=dict(t=50, l=25, r=25, b=25),
    width=900,
    height=500
)

fig.show()

# %% [markdown]
# ## distributions - working

# %%
import plotly.figure_factory as ff
import numpy as np

# Generate realistic price distributions for domestic appliances
np.random.seed(42)

# Washing machine prices (€300-€1200, with most around €600-€700)
washing_machine_prices = np.random.gamma(shape=8, scale=75, size=300) + 250

# Fridge prices (€400-€2000, with most around €800-€900)
fridge_prices = np.random.gamma(shape=6, scale=150, size=300) + 350

hist_data = [washing_machine_prices, fridge_prices]
group_labels = ['Washing Machines', 'Fridges']
colors = ['#3498DB', "#F58DC8"]

# Create distplot
fig = ff.create_distplot(
    hist_data, 
    group_labels, 
    show_hist=False,
    show_rug=True,
    colors=colors
)

# Update layout
fig.update_layout(
    title_text='Price Distribution of Domestic Appliances',
    xaxis_title='Price (€)',
    yaxis_title='Density',
    height=500,
    width=800,
    xaxis=dict(range=[0, 2500])
)

fig.show()

# %% [markdown]
# ## funnel blue  - working

# %%
import plotly.graph_objects as go

# Data
stages = ["Presentations", "Views", "Reads"]
values = [5800, 1200, 507]

# Create funnel chart
fig = go.Figure(go.Funnel(
    y = stages,
    x = values,
    textinfo = "value+percent initial"
))

# Update layout with custom width and height
fig.update_layout(
    title = "Medium Story Engagement Funnel",
    width = 600,   # Set custom width
    height = 500   # Set custom height
)

# Show the chart
fig.show()

# %% [markdown]
# ## funnel pink - working

# %%
import plotly.graph_objects as go

# Data
stages = ["Presentations", "Views", "Reads"]
values = [5800, 1200, 507]

# Create funnel chart with customizations
fig = go.Figure(go.Funnel(
    y = stages,
    x = values,
    textinfo = "value+percent previous",  # Show value and percent from previous stage
    textposition = "outside",              # Position text outside the bars
    marker = {
        "color": "hotpink" 
    },
    connector = {"fillcolor": "lightpink"} # Connector fill color
))

# Update layout with custom width and height
fig.update_layout(
    title = "Medium Story Engagement Funnel",
    width = 900,
    height = 500
)

# Show the chart
fig.show()


