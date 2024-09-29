import matplotlib.pyplot as plt
import pandas as pd
model_pairs = [
    ('CheapNet', 0.761, 0.870),
    ('ConBAP', 0.719, 0.864),
    ('ECIF', 0.672, 0.805),
    ('ECIFgraph', 0.556, 0.731),
    ('ECIF::HM', 0.711, 0.834),
    ('ECIF::LD-GBT', 0.711, 0.834),
    ('ECIFGraph::HM-Apo', 0.636, 0.772),
    ('ECIFGraph::HM-Holo', 0.645, 0.769),
    ('ECIFGraph::HM-Holo-Apo', 0.675, 0.820),
    ('AEScore', 0.640, 0.830),
    ('AK-score', 0.670, 0.812),
    ('HydraScreen', 0.672, 0.86),
    ('TopBP', -1, 0.861),
    ('HAC-Net', -1, 0.846),
    ('GIGN', 0.710, 0.840),
    ('PLANET', 0.682, 0.824),
    ('AGL-Score', -1, 0.833),
    (r'$K_{\mathregular{DEEP}}$', -1, 0.82),
    ('Pafnucy', -1, 0.78),
    ('OnionNet', -1, 0.816),
    ('OnionNet-2', -1, 0.864),
    ('IGN', 0.667, 0.821),
    (r'$\Delta_{\mathregular{vina}}\mathregular{RF}_{20}$', 0.635, 0.739),
    (r'$\Delta_{\mathregular{vina}}\mathregular{XGB}$', 0.647, 0.796),
    (r'$\Delta_{\mathregular{Lin\_F9}}\mathregular{XGB}$', 0.704, 0.845),
    ('EGNN', 0.660, 0.816),
    ('RTMScore', 0.531, 0.455),
    ('ChemPLP@GOLD', 0.633, 0.614),
    (r'$\mathregular{DrugScore}^{\mathregular{CSD}}$', 0.630, 0.596),
    ('LUDI2@DS', 0.629, 0.526),
    ('LUDI1@DS', 0.612, 0.494),
    ('LigScore2@DS', 0.608, 0.54),
    ('DrugScore2018', 0.607, 0.602),
    ('Affinity-dG@MOE', 0.604, 0.552),
    ('X-Score', 0.604, 0.629),
    ('X-ScoreHM', 0.603, 0.631),
    ('LigScore1@DS', 0.599, 0.425),
    ('ChemScore@SYBYL', 0.593, 0.59),
    ('London-dG@MOE', 0.593, 0.405),
    ('G-Score@SYBYL', 0.591, 0.572),
    ('PLP2@DS', 0.589, 0.563),
    (r'$\Delta \mathregular{SAS}$', 0.588, 0.625),
    ('PLP1@DS', 0.582, 0.581),
    ('D-Score@SYBYL', 0.577, 0.531),
    ('X-ScoreHP', 0.573, 0.609),
    ('ASP@GOLD', 0.553, 0.617),
    ('X-ScoreHS', 0.547, 0.621),
    ('PMF@DS', 0.537, 0.422),
    ('Alpha-HB@MOE', 0.535, 0.569),
    ('LUDI3@DS', 0.532, 0.502),
    ('Autodock Vina', 0.528, 0.604),
    ('ChemScore@GOLD', 0.526, 0.574),
    ('Jain@DS', 0.521, 0.457),
    ('GBVI_WSA-dG@MOE', 0.489, 0.496),
    ('PMF04@DS', 0.481, 0.212),
    ('PMF@SYBYL', 0.449, 0.262),
    ('ASE@MOE', 0.439, 0.591),
    ('GlideScoreSP', 0.419, 0.513),
    ('GoldScore@GOLD', 0.284, 0.416),
    ('GlideScore-XP', 0.257, 0.467)
]

# big font size
# plt.rcParams.update({'font.size': 20})
# Convert list to DataFrame
df = pd.DataFrame(model_pairs, columns=['Model', 'Spearman', 'Pearson'])

# Filtering
df_filtered_spearman = df[df['Spearman'] != -1]
df_filtered_pearson = df[df['Pearson'] != -1]

# Plot function
def plot_correlation_subplot(ax, df, correlation_column, title, top_n=None):
    if top_n:  # Plot only the top_n values if specified
        df = df.nlargest(top_n, correlation_column)
    df = df.sort_values(correlation_column)  # Sort values for plotting
    
    # Assign colors (Red for CheapNet, Gray for others)
    colors = ['red' if 'CheapNet' in model else 'gray' for model in df['Model']]
    
    # Create the horizontal bar chart
    bars = ax.barh(df['Model'], df[correlation_column], color=colors, edgecolor='black')
    
    # Add correlation values next to bars
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center', fontsize=16)
    
    # Set labels and title
    ax.set_xlabel(f'{correlation_column} Correlation Coefficient', fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.set_xlim(0, 1)
    
    # Remove grid for a cleaner look
    ax.grid(False)
    
    # Change tick params for the axis (font size)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

# fig, axes = plt.subplots(1, 2, figsize=(12, 3))
# plot_correlation_subplot(axes[0], df_filtered_spearman, 'Spearman', 'Top 10 Ranking Power', top_n=10)
# plot_correlation_subplot(axes[1], df_filtered_pearson, 'Pearson', 'Top 10 Scoring Power', top_n=10)
# plt.tight_layout()
# plt.savefig('figure/ranking_scoring_top10.png', format='png', bbox_inches='tight')
# plt.savefig('figure/ranking_scoring_top10.pdf', format='pdf', bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(18, 18))
plot_correlation_subplot(axes[0], df_filtered_spearman, 'Spearman', 'Ranking Power')
plot_correlation_subplot(axes[1], df_filtered_pearson, 'Pearson', 'Scoring Power')
plt.subplots_adjust(wspace=0.65)
plt.savefig('figure/ranking_scoring_full.png', format='png', bbox_inches='tight')
plt.savefig('figure/ranking_scoring_full.pdf', format='pdf', bbox_inches='tight')