import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

import logging
import yaml

logger = logging.getLogger(__name__)

def build_heatmap(input_path, output_path, feature_selection_method=None):
    """
    Build a heatmap visualization from chemical prediction data.
    
    Args:
        input_path (str or pathlib.Path): Path to the parquet file containing prediction data
        output_path (str or pathlib.Path): Path to save the output heatmap image and related files
    """
    # Convert paths to pathlib.Path objects if they're not already
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # tell pandas to never wrap, just keep going wider
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 100)
    
    logger.info(f"Building heatmap from {input_path}")
    
    # Load prediction data
    df_allfeat = pd.read_parquet(input_path)

    # filter feature
    feature_path = input_path.parent / 'selected_properties' / f'{feature_selection_method}_selected_properties.txt'
    feature_names = set(line.strip() for line in pathlib.Path(feature_path).read_text().splitlines() if line.strip())
    feature_names = sorted(list(feature_names))
    df_allfeat['is_in_lookup'] = df_allfeat['property_title'].isin(feature_names)
    df = df_allfeat[df_allfeat['is_in_lookup']]

    
    classdf = pd.read_csv(input_path.parent / 'classified_chemicals.csv') 

    if 'classification' not in df.columns or df['classification'].isna().any():
    # Merge classification info
        print('reading classification')
        df = df.drop(columns=['classification'], errors='ignore')  # drop to avoid _x/_y
        df = df[df['inchi'].isin(classdf['inchi'])] #filter inchi with classified label
        df = df.merge(classdf[['inchi', 'classification']], on='inchi', how='left') # 

    if 'name' not in df.columns:
        # print(df.columns)
        df['name'] = df['name_x']
        
    # Input for heatmap
    pdf = df[['name', 'property_token', 'value', 'classification']]


    # Filter data for ring compounds
    # pdf = pdf[pdf['classification'].str.contains("Ring")]
    
    # Generate the heatmap
    project_dir = input_path.parent / 'config.yaml'
    _generate_heatmap(pdf, output_path, project_dir)
    
    # Create a csv of the top 10 most activated properties
    top_props_path = output_path.parent / 'top_props.csv'
    top10df = df[['name', 'property_title', 'property_source', 'property_metadata', 'value', 'classification']]
    
    # str cant be used sometimes
    top10df = top10df[~top10df['classification'].str.contains("Paraffin")]
    
    top10_props = top10df\
        .groupby(['property_title', 'property_source', 'property_metadata'])['value'].mean()\
        .reset_index().sort_values('value', ascending=False)\
        .head(10)
    
    top10_props.to_csv(top_props_path, index=False)
    logger.info(f"Saved top properties to {top_props_path}")
    
    return output_path


def _generate_heatmap(pdf, output_path, project_dir):
    """
    Internal function to generate the actual heatmap visualization.
    
    Args:
        pdf (pandas.DataFrame): DataFrame containing the chemical prediction data
        output_path (pathlib.Path): Path to save the output heatmap image
        project_dir (pathlib.Path): Path to the project directory containing config.yaml
    """
    # Load project config
    config_path = pathlib.Path("config/projects") / f'{project_dir.name}.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    else:
        with open("config/default.yaml", "r") as f:
            config = yaml.safe_load(f)

    colors_hex = config['colors']
    color_threshold = config.get('heatmap', {}).get('color_threshold', 5)

    # Pivot and normalize
    pivot_df = pdf.pivot_table(index='name', columns='property_token', values='value', aggfunc='first')
    norm_values = pd.DataFrame(
        MinMaxScaler().fit_transform(pivot_df.fillna(0)), 
        index=pivot_df.index, 
        columns=pivot_df.columns
    )

    # Get unique classifications
    unique_classes = sorted(pdf['classification'].astype(str).unique())  # sorted for consistency
    # Safely assign only as many colors as needed
    category_colors = dict(zip(unique_classes, colors_hex[:len(unique_classes)]))

    # Apply classification and get row colors
    class_series = pdf.drop_duplicates('name').set_index('name')['classification']
    norm_values['substance_class'] = class_series
    row_colors = norm_values['substance_class'].map(category_colors)
    row_colors.name = None

    norm_values = norm_values.drop(columns='substance_class')

    # Thresholding dendrogram
    row_linkage = sch.linkage(pdist(norm_values, metric='euclidean'), method='average')

    # Plot heatmap with adjusted layout - keep column clustering
    sns.set(style="white")

    # if the map shows black, reduce linewidths.
    g = sns.clustermap(
        norm_values, cmap="viridis",
        row_colors=row_colors, row_linkage=row_linkage,
        xticklabels=False, yticklabels=True, 
        # linewidths=0.001,
        linecolor='black', col_cluster=True, row_cluster=True,
        figsize=(18, 9),  # Wider figure
        cbar_pos=(0.91, 0.3, 0.02, 0.4),  # More to the left and higher up
        dendrogram_ratio=(0.1, 0.05),  # Make column dendrogram shorter (was 0.2 by default)
        tree_kws={'linewidths': 0.5}  # Thinner dendrogram lines
    )
    plt.show()
    reordered_ind = g.dendrogram_row.reordered_ind
    row_labels = g.data.index[reordered_ind]
    
    color_threshold = 3 # for now.
    cluster_ids = sch.fcluster(row_linkage, t=color_threshold, criterion='distance')
    cluster_ids_reordered = cluster_ids[reordered_ind]
    norm_values_reordered = g.data.iloc[reordered_ind]
    row_means = norm_values_reordered.mean(axis=1)
    row_medians = norm_values_reordered.median(axis=1)

    gdata_with_class = g.data.copy()
    gdata_with_class['classification'] = class_series
    gdata_with_class_reordered = gdata_with_class.iloc[reordered_ind]

    # Save to CSV
    df = pd.DataFrame({
        'row_label': row_labels,
        'substance_class': gdata_with_class_reordered['classification'].values,
        'cluster_id': cluster_ids_reordered,
        'mean': row_means.values,
        'median': row_medians.values
    })    
    df.to_csv(output_path.parent / 'heatmap_labels.csv', index=False, header=False)    # Hide column dendrogram but still keep clustering
    g.ax_col_dendrogram.set_visible(False)

    # Now manually color the dendrogram using scipy's dendrogram and the same axes
    ax = g.ax_row_dendrogram
    # Clear the default dendrogram
    ax.clear()
    # Redraw dendrogram using scipy with your threshold
    sch.dendrogram(row_linkage, ax=ax, orientation='left', color_threshold=color_threshold, above_threshold_color='gray')

    #11 for nephro
    ax.invert_yaxis()

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    # plt.show()
    left_margin = 0.10  # Amount of padding on the left

    # Shift heatmap
    heatmap_pos = g.ax_heatmap.get_position()
    g.ax_heatmap.set_position([
        heatmap_pos.x0 + left_margin, heatmap_pos.y0,
        heatmap_pos.width * 0.88, heatmap_pos.height
    ])

    # Shift row dendrogram
    row_pos = g.ax_row_dendrogram.get_position()
    g.ax_row_dendrogram.set_position([
        row_pos.x0 + left_margin, row_pos.y0,
        row_pos.width, row_pos.height
    ])

    # Shift row colors (if exists)
    if hasattr(g, 'ax_row_colors'):
        row_colors_pos = g.ax_row_colors.get_position()
        g.ax_row_colors.set_position([
            row_colors_pos.x0 + left_margin, row_colors_pos.y0,
            row_colors_pos.width, row_colors_pos.height
    ])


    # Create a separate axis for the class legend on the left (moved further left)
    # left_legend_ax = plt.axes([0.005, 0.7, 0.05, 0.2])  # [x, y, width, height] - moved x from 0.01 to 0.005
    left_legend_ax = plt.axes([0.01, 0.3, 0.04, 0.3])
    left_legend_ax.axis('off')  # Hide the axis

    # Add class legend to the left axis
    class_legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in category_colors.values()]
    legend = left_legend_ax.legend(
        class_legend_handles, 
        category_colors.keys(), 
        loc='center', 
        frameon=False, 
        title='Class', 
        title_fontsize=20,
        labelcolor='white',
        prop={'size': 20}
        # # handlelength=1.5,                # Length of rectangle
        # handleheight=1.5,                # Height spacing
        # handletextpad=0.8                # Padding between rectangle and label
    )
    legend.get_title().set_color('white')

    # Adjust the colorbar (scale legend) that's now on the far right
    cbar = g.ax_cbar
    # cbar.set_ylabel('Scale', color='white', fontsize=20)
    cbar.set_xlabel('Scale', color='white', fontsize=20, labelpad=10)
    cbar.tick_params(colors='white', labelsize=20)

    cbar_pos = cbar.get_position()
    cbar.set_position([
        cbar_pos.x0 + left_margin,  # shift right by same margin or more
        cbar_pos.y0,
        cbar_pos.width,
        cbar_pos.height
    ])
    
    # Add axis labels but no title
    g.ax_heatmap.set_xlabel("Estimated property values", color='white', fontsize=20)
    g.ax_heatmap.set_ylabel("Substance", color='white', fontsize=20)
    
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved heatmap to {output_path}")