import pandas as pd
import os

def get_season(month):
    """Returns the season based on the month number."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def process_csv(file_path):
    try:
        # Load the CSV
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Create the deviation column
        df['Lab 409 Room Temp Deviation From Setpoint (°F)'] = abs(
            df['Lab 409 Room Temp (°F)'] - df['Lab 409 Room Temp Setpoint (°F)']
        )
        
        # Drop unnecessary columns
        columns_to_drop = [
            "Lab 409 Room Temp Setpoint (°F)", 
            "Final Fault Status", 
            "Reheat Valve Command (-)"
        ]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        
        # Extract temporal features
        df['Hour of the Day'] = df['Date'].dt.hour
        df['Day of the Week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
        df['Season'] = df['Date'].dt.month.apply(get_season)
        
        # Save the processed file with the same name as input
        df.to_csv(file_path, index=False)
        print(f"Processed file saved as: {file_path}")
        
    except Exception as e:
        print(f"Error processing the file: {e}")

def combine_csv(file1, file2, output_file):
    try:
        # Load both CSVs
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Align columns (ensure they have the same order)
        common_columns = list(set(df1.columns) & set(df2.columns))
        df1 = df1[common_columns]
        df2 = df2[common_columns]
        
        # Combine the datasets
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Sort the combined dataset by Date in chronological order
        combined_df.sort_values(by='Date', inplace=True)
        
         # Reorder columns to make 'Date' the first column and 'Label' the last column if it exists
        columns = list(combined_df.columns)
        columns.remove('Date')
        if 'Label' in columns:
            columns.remove('Label')
            column_order = ['Date'] + columns + ['Label']
        else:
            column_order = ['Date'] + columns

        combined_df = combined_df[column_order]

        
        # Save the merged file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined file saved as: {output_file}")
        
    except Exception as e:
        print(f"Error combining files: {e}")

if __name__ == "__main__":
    option = input("Enter 1 to process a CSV, 2 to combine two CSVs: ").strip()
    
    if option == "1":
        file_path = input("Enter the path to the CSV file: ").strip()
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            process_csv(file_path)
        else:
            print("Invalid file path. Please ensure the file exists and is a CSV.")
    elif option == "2":
        file1 = input("Enter the path to the first CSV file: ").strip()
        file2 = input("Enter the path to the second CSV file: ").strip()
        output_file = input("Enter the path for the output CSV file: ").strip()
        
        if all(os.path.exists(f) and f.endswith('.csv') for f in [file1, file2]):
            combine_csv(file1, file2, output_file)
        else:
            print("Invalid file paths. Please ensure both files exist and are CSVs.")
