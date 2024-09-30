import subprocess

def run_clustering(input_fasta, output_dir, tmp_dir, min_seq_id=0.9, cov=0.8):
    try:
        subprocess.run([
            "mmseqs", "easy-cluster",
            input_fasta,
            f"{output_dir}/cluster-results",
            tmp_dir,
            "--min-seq-id", str(min_seq_id),
            "-c", str(cov),
            "--threads", "8",
            "--cov-mode", "1",
        ], check=True)
        print("clustering completed successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error running mmseqs: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")