import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops, Descriptors
import warnings
import logging
import os
import pathlib

# Suppress RDKit warnings
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')


def classify_molecule(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return "Error: RDKit Failed to Parse InChI"

        # Get the number of rings in the molecule
        # GetSSSR returns the Smallest Set of Smallest Rings
        # which is more accurate for chemical classification
        num_rings = len(rdmolops.GetSSSR(mol))
        num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        logging.info(f"Number of rings: {num_rings}")

        # Check for Paraffins (acyclic CnH(2n+2))
        if num_rings == 0:
            # Check if molecule only contains C and 	H atoms
            only_c_h = all(atom.GetAtomicNum() in [1, 6] for atom in mol.GetAtoms())
            num_hydrogens_implicit = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

            if only_c_h and num_hydrogens_implicit == (2 * num_carbons + 2):
                return f"C{num_carbons} Paraffin"

            return "Acyclic (Non-Paraffin)"

        # Classify by ring count
        if num_rings <= 5:
            return f"{num_rings} Ring Aromatic"
        return f"{num_rings}+ Ring Aromatic"  # 6+ rings

    except Exception as e:
        logging.error(f"Error classifying molecule: {str(e)}")
        return f"Error: RDKit Classification Error ({type(e).__name__})"


def categorize_chemicals(input_path, output_path):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(output_path, 'log.txt'), level=logging.INFO, filemode='w')
    # Read the CSV file
    df = pd.read_csv(input_path)

    chemicals_csv = input_path.parent / 'chemicals.csv'
    
    # Add classification column
    if chemicals_csv.exists():
        indf = pd.read_csv(chemicals_csv)
        indf.columns = indf.columns.str.lower()
        indf = indf[['label','name']].copy()

        df = df.merge(indf, on='name', how='left')
        # TO-DO: config
        df['classification'] = df['label'].map({0: 'Non-toxic', 1: 'Toxic'})
        # df['classification'] = df['label']#.map({0: 'Non-DNT', 1: 'DNT'})
        # df['classification'] = df['label'].map({0: 'Non-nephrotoxic', 1: 'Nephrotoxic'})
    else:
        df['classification'] = df['inchi'].apply(classify_molecule)

    # Save results
    df.to_csv(output_path / 'classified_chemicals.csv', index=False)

    # Print summary
    logging.info("\nClassification Summary:")
    logging.info(df['classification'].value_counts())

    return df



