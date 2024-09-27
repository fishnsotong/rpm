# python preprocessing.py --input data/raw --output data/processed --url https://rna.urmc.rochester.edu/pub/archiveII.tar.gz

import os
import argparse
import requests
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split

from clustering import run_clustering

def download_data(url: str, output_dir: str):
    """
    Download raw data from a given URL and save it to the specified directory.

    :param url: URL to download the data from
    :param output_dir: Directory to save the downloaded data
    """
    # download the file, define file name based on URL
    print(f"Downloading data from {url} to {output_dir}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Failed to download data: {e}")
        return

    tar_file_name = os.path.join(os.getcwd(), url.split("/")[-1])

    with open(tar_file_name, 'wb') as f:
        f.write(r.content)

    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # extract
    with tarfile.open(tar_file_name, 'r:gz') as tar:
        tar.extractall(path=output_dir)
    
    # remove the downloaded tar file after extraction
    os.remove(tar_file_name)

    print(f"Downloaded and extracted {tar_file_name} to {output_dir}")
    pass

def find_ct_files(input_dir: str) -> list:
    """
    Finds all .ct files in the given directory.
    
    :param input_dir: The directory where to look for CT files.
    :return: A list of paths to CT files.
    """
    # return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".ct")]
    return [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.endswith(".ct")]

def extract_rna_sequence_from_ct(ct_file: str) -> tuple[str, str]:
    """
    Extracts the RNA sequence from a CT file. Evaluates if a pseudoknot 
    :param ct_file: Path to a CT file.
    :return: A tuple containing the filename (without extension) and the RNA sequence.
    """
    sequence = []
    with open(ct_file, 'r') as file:
        lines = file.readlines()
        # Skip the header line, process each nucleotide line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            nucleotide = parts[1]
            sequence.append(nucleotide)
    filename = os.path.splitext(os.path.basename(ct_file))[0]
    # Return the filename without extension and the sequence
    return filename, ''.join(sequence)

def ct_to_fasta(input_dir: str, output_dir: str) -> str:
    """
    Finds all CT files in the input directory, extracts their RNA sequences,
    and saves them in the specified FASTA file.

    TODO: add support for storing DBN structures in the FASTA file
          instead of outputting FASTA file, it should be one big DBN file?

    :param input_dir: Directory containing CT files.
    :param output_fasta_file: Path to the output FASTA file.
    """
    # find all ct files in current directory
    ct_files = find_ct_files(input_dir)

    # build path for output file
    output_fasta_file = os.path.join(output_dir, "rna_sequences.fasta")
    
    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(output_fasta_file, 'w') as fasta_file:
        for ct_file in ct_files:
            count += 1
            filename, sequence = extract_rna_sequence_from_ct(ct_file)
            # Write the sequence in FASTA format 
            fasta_file.write(f">{filename}\n")
            fasta_file.write(f"{sequence}\n")
            fasta_file.write("\n")
    print(f"Processed {count} CT file(s)!")

    # return the path to the output file, for further processing
    return output_fasta_file

def fasta_parse(fasta_file: str, comment='#'):
    """
    Parses a FASTA file and extracts sequence names and their corresponding sequences.

    The function reads a given FASTA file, where each sequence is identified by a line starting with 
    the '>' symbol, followed by the sequence name. The subsequent lines contain the sequence data, 
    which is concatenated into a single string for each sequence. The function also ignores any comment 
    lines that start with the specified comment character.

    Parameters:

    fasta_file : str
        The path to the FASTA file to be parsed.
    comment : str, optional
        A character indicating comment lines that should be ignored. Default is '#'.
    
    Returns:
    -------
    tuple:
        A tuple containing two lists:
        - names : List[str]
            A list of sequence names extracted from lines starting with '>'.
        - sequences : List[str]
            A list of corresponding sequences, with each sequence represented as a string.
    
    Example:
    --------
    >>> names, sequences = fasta_parse("example.fasta")
    >>> print(names)
    ['sequence1', 'sequence2']
    >>> print(sequences)
    ['ATCGATCG', 'GGCTAAGT']

    Notes:
    ------
    - Sequence names are taken from lines starting with '>', with the '>' character removed.
    - Sequences are converted to uppercase.
    - The function assumes that the sequences are stored in a standard FASTA format.

    """
    names = []
    sequences = []
    name = None
    sequence = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(comment):
                continue
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    names.append(name)
                    sequences.append(''.join(sequence))
                name = line[1:]
                sequence = []
            else:
                sequence.append(line.upper())
        if name is not None:
            names.append(name)
            sequences.append(''.join(sequence))

    return names, sequences

def dedup_sequences(data_tuple) -> tuple[list[str], list[str]]:
    """
    Removes duplicate sequences from a tuple containing sequence names and sequences.

    This function takes a tuple consisting of two lists: one for sequence names and 
    one for the corresponding sequences. It removes duplicate sequences and returns 
    a new tuple containing only the unique sequences and their corresponding names.

    Parameters:
    ----------
    data_tuple : tuple
        A tuple containing two lists:
        - names (list of str): A list of sequence names.
        - sequences (list of str): A list of nucleotide or protein sequences.

    Returns:
    -------
    tuple
        A tuple containing two lists:
        - result_names (list of str): A list of names corresponding to the unique sequences.
        - result_sequences (list of str): A list of unique sequences.

    Example:
    --------
    >>> names = ["seq1", "seq2", "seq3"]
    >>> sequences = ["AGCT", "CGTA", "AGCT"]
    >>> data_tuple = (names, sequences)
    >>> dedup_sequences(data_tuple)
    (["seq1", "seq2"], ["AGCT", "CGTA"])

    This function can also be modified to produce a tuple of duplicate values.
    """
    names, sequences = data_tuple
    unique_sequences = {}
    result_names = []
    result_sequences = []
    counter = 0
    
    for name, seq in zip(names, sequences):
        if seq not in unique_sequences:
            unique_sequences[seq] = name
            result_names.append(name)
            result_sequences.append(seq)
        else:
            print(f"Duplicate sequence found: {name} and {unique_sequences[seq]}")
            counter += 1

    print(f"\nTotal duplicate sequences found: {counter}")
    return (result_names, result_sequences)

def fasta_write(data_tuple, output_fasta) -> tuple:
    """
    Writes a tuple of names and sequences to a FASTA file.
    
    Parameters:
    ----------
    data_tuple : tuple
        A tuple where the first element is a list of names and the second element is a list of sequences.
    output_fasta : str
        The name of the output FASTA file.
    """
    names, sequences = data_tuple
    with open(output_fasta, 'w') as fasta_file:
        for name, sequence in zip(names, sequences):
            fasta_file.write(f">{name}\n{sequence}\n")
    print(f"Data written to {output_fasta}")
    
    return data_tuple

def ct2dbn(ct_filename: str) -> str:
    """
    Converts a CT (Connectivity Table) file into a dot-bracket notation (DBN) string representing RNA 
    secondary structure.

    This function parses a CT file to extract the RNA sequence and base-pairing information, 
    then generates a corresponding dot-bracket notation (DBN) string. The DBN string uses 
    various types of brackets to indicate base pairs, while unpaired nucleotides are denoted by dots (`.`).

    Parameters:
    ----------
    ct_filename : str
        Path to the CT file that contains RNA sequence and base-pairing information.
        The CT file format contains nucleotide information and the indices of base-pairing partners.

    Returns:
    -------
    str
        A string in dot-bracket notation representing the RNA secondary structure.
        The string consists of dots (unpaired nucleotides) and various brackets (paired nucleotides).
        Different types of brackets (e.g., '()', '<>', '[]', '{}') represent nested and pseudoknotted structures.

    Example:
    --------
    >>> ct2dbn('example.ct')
    '..((..<<..>>..))..'

    Notes:
    ------
    - The CT file typically contains columns representing the nucleotide index, the nucleotide itself, 
      and the index of its paired nucleotide (or `0` if unpaired).
    - Pseudoknots and nested structures are represented by different levels of brackets ('()', '<>', '[]', '{}').
    - The function handles base-pairing information, deduplicates pairs, and assigns brackets at different 
      nesting levels.
    - If a pseudoknot or interleaving base pair is detected, the function advances to the next available 
      bracket level.
    - The number of bracket levels is limited by the predefined list (`levels`), which currently supports 
      up to four levels.

    """
    name = None
    sequence = []
    raw_pairlist = []

    with open(ct_filename, 'r') as f:
        for i, line in enumerate(f):
            if not i:
                name = line.split()[1]              # name of the ncRNA
            else:
                sequence.append(line.split()[1])    # save primary sequence to string

                n = int(line.split()[0])            # nucleotide index
                k = int(line.split()[4])            # base-pairing partner index
                
                # only considered paired bases (k != 0)
                if k > 0:
                    raw_pairlist.append((n, k))

    # deduplicate pairlist
    pairlist = []
    seen = set()

    for pair in raw_pairlist:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in seen:
            seen.add(sorted_pair)
            pairlist.append(sorted_pair)

    dots = ['.'] * len(sequence)

    levels = ['()', '<>', '[]', '{}']
    current_level = 0

    for i, (start, end) in enumerate(pairlist):
        for j, (prev_start, prev_end) in enumerate(pairlist[:i]):
            # Is there any interleaving with previous pairs?
            if prev_start < start < prev_end < end:

                # Control advancement of the current_level but ensure it does not exceed the highest 
                # allowable level (determined by the length of the levels list).
                current_level = min(current_level + 1, len(levels) - 1)
                break
            else:
                current_level = 0

            # Use the current level's brackets for this base pair
        open_bracket, close_bracket = levels[current_level]
        if dots[start - 1] == "." and dots[end - 1] == ".":
            dots[start - 1] = open_bracket
            dots[end - 1] = close_bracket

    return "".join(dots)

def pseudoknot_checker(dbn: str) -> int:
    """
    Checks for the presence of pseudoknots in a dot-bracket notation (DBN) string.

    This function scans a dot-bracket notation (DBN) string to detect the presence of pseudoknots.
    A pseudoknot is typically indicated by the presence of angle brackets (`<` or `>`), which represent
    non-canonical base pairing interactions that are not nested within the usual dot-bracket structure.

    Parameters:
    ----------
    dbn : str
        A string representing RNA secondary structure in dot-bracket notation.
        Valid characters include '.', '(', ')', '[', ']', '<', '>', etc.

    Returns:
    -------
    int
        Returns 1 if pseudoknots (angle brackets '<' or '>') are detected in the DBN string, otherwise returns 0.

    Example:
    --------
    >>> dbn = "..((..<<..>>..)).."
    >>> pseudoknot_checker(dbn)
    1
    
    >>> dbn = "..((..)).."
    >>> pseudoknot_checker(dbn)
    0
    """
    pseudoknot_state = 0
    for i in dbn:
        # if "<" in dbn: 
        # even if there's a for loop in the background it's still more efficient
        # TODO: return "<" in dbn 
        if i in "<>":
            pseudoknot_state = 1
            break

    return pseudoknot_state

def add_labels_to_data(data_tuple, input_dir) ->tuple:
    """
    Adds classification labels to the RNA sequences in the provided data tuple based on pseudoknot detection.

    This function processes RNA sequence names, searches for corresponding .ct files in the specified input 
    directory, and assigns labels to the sequences. The labels are generated by detecting pseudoknots in 
    the RNA structures. If a corresponding .ct file is not found for a sequence, a default label of 0 is 
    assigned.

    Parameters
    ----------
    data_tuple : tuple
        A tuple containing two elements:
        - names (list): A list of RNA sequence names.
        - sequences (list): A list of RNA sequences corresponding to the names.
        
    input_dir : str
        The directory where .ct files are located. The function will search for the .ct files that match 
        the RNA sequence names to generate labels.

    Returns
    -------
    tuple
        A tuple containing:
        - names (list): The original list of RNA sequence names.
        - sequences (list): The original list of RNA sequences.
        - labels (list): A list of labels where each label is determined by pseudoknot detection 
          (e.g., using the `pseudoknot_checker`). If a corresponding .ct file is not found, a label 
          of 0 is assigned.
    """
    names, sequences = data_tuple
    labels = []

    # Create a dictionary to map filenames to their paths
    ct_file_paths = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".ct"):
                ct_file_paths[os.path.splitext(file)[0]] = os.path.join(root, file)

    for name in names:
        ct_file_path = ct_file_paths.get(name)
        if ct_file_path:
            labels.append(pseudoknot_checker(ct2dbn(ct_file_path)))
        else:
            labels.append(0)  # or handle the case where the file is not found

    return (names, sequences, labels)

def train_val_test_split(data_tuple, cluster_df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits the data into train, validation, and test sets, ensuring that sequences 
    from the same cluster are only present in one of the sets.

    Parameters:
    ----------
    data_tuple : tuple
        A tuple where the first element is a list of names, the second element is a list of sequences,
        and the third element is a list of classification labels.
    cluster_df : pd.DataFrame
        A DataFrame with two columns: 'Cluster_Rep' and 'Cluster_Member', where 'Cluster_Member'
        corresponds to names in the data_tuple.
    test_size : float
        Proportion of the data to include in the test split.
    val_size : float
        Proportion of the training data to include in the validation split.
    random_state : int or None
        Random seed for reproducibility.
        
    Returns:
    -------
    train_set : tuple
        A tuple containing names, sequences, and labels for the training set.
    val_set : tuple
        A tuple containing names, sequences, and labels for the validation set.
    test_set : tuple
        A tuple containing names, sequences, and labels for the test set.
    """

    # Extract names, sequences, and labels from data_tuple
    names, sequences, labels = data_tuple

    # Map names to sequences and labels
    name_to_seq = {name: seq for name, seq in zip(names, sequences)}
    name_to_label = {name: label for name, label in zip(names, labels)}

    # Get unique clusters
    clusters = cluster_df['Cluster_Rep'].unique()

    # Split clusters into train and test sets
    train_clusters, test_clusters = train_test_split(
        clusters, test_size=test_size, random_state=random_state
    )

    # Further split the training clusters into train and validation sets
    train_clusters, val_clusters = train_test_split(
        train_clusters, test_size=val_size, random_state=random_state
    )

    # Get names for each set by filtering the cluster_df
    train_names = cluster_df[cluster_df['Cluster_Rep'].isin(train_clusters)]['Cluster_Member'].tolist()
    val_names = cluster_df[cluster_df['Cluster_Rep'].isin(val_clusters)]['Cluster_Member'].tolist()
    test_names = cluster_df[cluster_df['Cluster_Rep'].isin(test_clusters)]['Cluster_Member'].tolist()

    # Retrieve sequences and labels for the names in each set
    train_set = (
        [name for name in train_names], 
        [name_to_seq[name] for name in train_names],
        [name_to_label[name] for name in train_names]
    )
    val_set = (
        [name for name in val_names], 
        [name_to_seq[name] for name in val_names],
        [name_to_label[name] for name in val_names]
    )
    test_set = (
        [name for name in test_names], 
        [name_to_seq[name] for name in test_names],
        [name_to_label[name] for name in test_names]
    )

    return train_set, val_set, test_set

def main():
    # TODO: set up logging

    # set up argument parsing  
    parser = argparse.ArgumentParser(description="Download and preprocess data")
    parser.add_argument("--url", type=str, required=False, help="URL to download data from")
    parser.add_argument("--input", type=str, required=True, help="Path to directory where raw data is stored")
    parser.add_argument("--output", type=str, required=True, help="Path to the directory where preprocessed data will be stored")
    args = parser.parse_args()

    # download the data if it does not exist
    if not os.listdir(args.input):
        download_data(args.url, args.input)
    else:
        print(f"Data already exists in {args.input}. Skipping download.")

    # convert CT files to FASTA format
    fasta_file = ct_to_fasta(args.input, args.output)

    # remove duplicate sequences
    data = fasta_write(dedup_sequences(fasta_parse(fasta_file)), os.path.join(args.output, "dedup_rna_sequences.fasta"))
    data = add_labels_to_data(data, args.input)

    # mmseqs clustering
    run_clustering(input_fasta=os.path.join(args.output, "dedup_rna_sequences.fasta"), output_dir=args.output, tmp_dir="./tmp")

    # load cluster data
    cluster_df = pd.read_csv(os.path.join(args.output, "cluster-results_cluster.tsv"), sep="\t",
                             header=None, names=['Cluster_Rep', 'Cluster_Member'])

    # train, validation, test split
    train_set, val_set, test_set = train_val_test_split(data, cluster_df, 
                                                    test_size=0.2, val_size=0.2, random_state=42)

    print("Training set:",len(train_set[0]))
    print("Validation set:", len(val_set[0]))
    print("Test set:", len(test_set[0]))

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()