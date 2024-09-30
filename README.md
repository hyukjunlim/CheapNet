
# CheapNet: Cross-attention on Hierarchical representations for Efficient protein-ligand binding Affinity Prediction

We propose CheapNet, a novel interaction-based model that integrates atom-level representations with hierarchical cluster-level interactions through a cross-attention mechanism. By employing differentiable pooling of atom-level embeddings, CheapNet efficiently captures essential higher-order molecular representations crucial for accurate binding predictions. Extensive evaluations demonstrate that CheapNet not only achieves state-of-the-art performance across multiple binding affinity prediction tasks but also maintains prediction accuracy with reasonable computational efficiency.

## key contributions of CheapNet

  - We propose a hierarchical model that integrates atom-level and cluster-level interactions, improving the representation of protein-ligand complexes.
  - Our model incorporates a cross-attention mechanism between protein and ligand clusters, focusing on biologically relevant binding interactions.
  - CheapNet achieves state-of-the-art performance across multiple binding affinity prediction tasks while maintaining computational efficiency.

## Key Features

- **Hierarchical Representations**: Uses higher-level node and cluster representations to enhance the understanding of protein-ligand interactions.
- **Cross-Attention Mechanism**: Leverages cross-attention to capture significnat interactions between protein and ligand clusters.
- **Efficiency**: Designed to be memory-efficient, requiring minimal memory and computation compared to other models.
- **Generalizability**: Achieves high accuracy of protein-ligand binding affinity prediction across various datasets.

## Installation

To install and use CheapNet, follow these steps:

1. Clone the repository:
   ```bash
   git clone 
   cd cheapnet
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up the environment using Conda:
   ```bash
   conda create --name cheapnet_env python=3.8
   conda activate cheapnet_env
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train CheapNet on your dataset, modify the data paths in `config.py` and run the following command:

```bash
python train.py --config config.yaml
```

### Prediction

For predicting the binding affinity of protein-ligand complexes, use the `predict.py` script:

```bash
python predict.py --input data/sample_input.csv --output results.csv
```

### Evaluation

To evaluate the model's performance, use the evaluation scripts provided:

```bash
python evaluate.py --dataset CASF2016
```

### Memory Footprint Analysis

A detailed memory footprint analysis can be performed to compare CheapNet's efficiency with other attention-based models:

```bash
python memory_analysis.py
```

## Datasets

- **CASF-2016**: The model has been benchmarked on the CASF-2016 dataset, showing competitive performance.
- **Cross-Dataset Evaluation**: CheapNet generalizes across multiple datasets using GIGN for cross-dataset evaluation, with overlap reduction following Atom3D protocols.

## Results

- **Spearman Correlation**: CheapNet closely follows $\Delta$vinaRF20 in ranking power for the CASF-2016 dataset, marking a significant advancement in this area.
- **Case Study**: Analysis on specific protein-ligand complexes (e.g., PDB ID: 4kz6) highlights CHEAPNet’s ability to capture key binding regions through its cluster-level representation and cross-attention mechanisms.

## Citation

If you find CheapNet useful in your research, please cite:

```
@article{yourname2024cheapnet,
  title={CheapNet: Cross-attention on Hierarchical Representations for Enhanced Protein-Ligand Binding Affinity Prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  volume={X},
  number={X},
  pages={X--X}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, features, or questions.

## Contact

For any inquiries or collaboration opportunities, please contact `your.email@domain.com`.
