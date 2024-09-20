import matplotlib.pyplot as plt

# Data for the plot
models = ['ΔvinaRF20', 'CheapNet', 'GIGN', 'PLANET', 'IGN', 'EGNN', 'ChemPLP@GOLD', 'DrugScoreCSD', 'LUDI2@DS', 'LUDI1@DS']
spearman_coefficients = [0.750, 0.744, 0.709, 0.682, 0.660, 0.649, 0.633, 0.630, 0.629, 0.612]

# # Data for the plot
# models = ['ΔvinaRF20', 'CheapNet', 'GIGN', 'PLANET', 'IGN', 'EGNN', 'ChemPLP@GOLD', 'DrugScoreCSD', 'LUDI2@DS', 'LUDI1@DS', 
#           'LigScore2@DS', 'DrugScore2018', 'Affinity-dG@MOE', 'X-Score', 'X-ScoreHM', 
#           'LigScore1@DS', 'ChemScore@SYBYL', 'London-dG@MOE', 'G-Score@SYBYL', 'PLP2@DS', 'ΔSAS', 'PLP1@DS', 
#           'D-Score@SYBYL', 'X-ScoreHP', 'ASP@GOLD', 'X-ScoreHS', 'PMF@DS', 'Alpha-HB@MOE', 'LUDI3@DS', 
#           'Autodock Vina', 'ChemScore@GOLD', 'Jain@DS', 'GBVI/WSA-dG@MOE', 'PMF04@DS', 'PMF@SYBYL', 'ASE@MOE', 
#           'GlideScoreSP', 'GoldScore@GOLD', 'GlideScore-XP']
# spearman_coefficients = [0.750, 0.744, 0.709, 0.682, 0.660, 0.649, 0.633, 0.630, 0.629, 0.612, 0.608, 0.607, 
#                          0.604, 0.604, 0.603, 0.599, 0.593, 0.593, 0.591, 0.589, 0.588, 
#                          0.582, 0.577, 0.573, 0.553, 0.547, 0.537, 0.535, 0.532, 0.528, 0.526, 0.521, 0.489, 0.481, 
#                          0.449, 0.439, 0.419, 0.284, 0.257]

# Create the figure with higher DPI and set size for paper format
plt.figure(figsize=(6, 4), dpi=300)
colors = ['red' if 'CheapNet' in model else 'gray' for model in models[::-1]]
bars = plt.barh(models[::-1], spearman_coefficients[::-1], color=colors, edgecolor='black')

# Add Spearman correlation values next to the bars
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.3f}', va='center', fontsize=10)
# Set labels and title with appropriate font for academic paper
plt.xlabel('Spearman Correlation Coefficient', fontsize=12)
plt.title('Top 10 Ranking Power', fontsize=14)
plt.xlim(0, 0.85)

# Remove the grid for a cleaner look
plt.grid(False)

# Use tight layout and adjust bbox_inches to ensure text fits
plt.tight_layout()
plt.savefig('spearman_correlation_paper_figure_top10.png', format='png', bbox_inches='tight')
plt.savefig('spearman_correlation_paper_figure_top10.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
