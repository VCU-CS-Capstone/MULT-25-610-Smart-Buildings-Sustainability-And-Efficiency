import pandas as pd

# Load both datasets
fault_df = pd.read_csv("test_data_f_labeled.csv")
normal_df = pd.read_csv("test_data_n_labeled.csv")

# Sample a portion from each
fault_sample = fault_df.sample(n=300, random_state=42)  
normal_sample = normal_df.sample(n=300, random_state=42)

# Combine and shuffle
mixed_df = pd.concat([fault_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
mixed_df.to_csv("test_data_mixed.csv", index=False)

print("Mixed test dataset saved as 'test_data_mixed.csv'")
