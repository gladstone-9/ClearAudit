from snsynth import Synthesizer
from snsynth.transform import MinMaxTransformer, TableTransformer
import pandas as pd
import sys
import pickle


# df: Pandas Data Frame
def generate_synthetic_data(df, needed_epsilon):
    preprocessor_eps = 1.5
    max_attempts = 10

    # min/max values per column
    transformers = []
    for column in df.columns:
        min_value = df[column].min()
        max_value = df[column].max()
        transformers.append(MinMaxTransformer(lower=min_value, upper=max_value))
        
    tt = TableTransformer(transformers)
    
    # Try generating synthetic data with increasing epsilon if it fails
    attempt = 0
    while attempt < max_attempts:
        try:
            synth = Synthesizer.create("mwem", epsilon=(preprocessor_eps + needed_epsilon), split_factor=5, verbose=True)
            
            synthetic_df = synth.fit_sample(
                df,
                transformer=tt,
                preprocessor_eps=preprocessor_eps,
            )
            return synthetic_df  # Return on success
        except Exception as e:
            attempt += 1
            preprocessor_eps += 0.5  # Increase epsilon if it fails
            print(f"Attempt {attempt} failed: {e}. Retrying with preprocessor_eps = {preprocessor_eps}...")

    # If all attempts fail, raise an error
    raise Exception("Failed to generate synthetic data after multiple attempts.")


# Test
# data = pd.read_csv("examples/wine.csv", index_col=None)
# df = generate_synthetic_data(data, 3)

# print(df.head(10))

if __name__ == "__main__":
    # Read input file paths from arguments
    input_df_path = sys.argv[1]
    needed_epsilon = float(sys.argv[2])
    output_df_path = sys.argv[3]
    
    # Load the input DataFrame
    with open(input_df_path, "rb") as f:
        df = pickle.load(f)

    # Run synthetic generation
    synthetic_df = generate_synthetic_data(df, needed_epsilon)

    # Save the output DataFrame
    with open(output_df_path, "wb") as f:
        pickle.dump(synthetic_df, f)
    
    # print(synthetic_df.head(10))
    
    sys.exit(0)