import pandas as pd
import glob
import json

def parquet_to_json(parquet_file, json_file):
  """
  Converts a Parquet file to a JSON file.

  Args:
    parquet_file: Path to the Parquet file.
    json_file: Path to the output JSON file.
  """
  try:
    df = pd.read_parquet(parquet_file)
    df.to_json(json_file, orient='records', lines=True)
    print(f"Successfully converted {parquet_file} to {json_file}")
  except Exception as e:
    print(f"Error converting {parquet_file}: {e}")

wiki40b_en_dir = "wiki40b/en/"
wiki40b_save_dir = "wiki40b/all_data/"

files = sorted(glob.glob(str(wiki40b_en_dir+"t*.parquet")))
for i, file in enumerate(files):
    parquet_to_json(file, str(wiki40b_en_dir+"train" + str(i) + ".json"))

files = sorted(glob.glob(str(wiki40b_en_dir+"v*.parquet")))
for i, file in enumerate(files):
    parquet_to_json(file, str(wiki40b_en_dir+"validation" + str(i) + ".json"))


def convert_multiline_json_to_single_line(input_filepath, output_filepath):
    """
    Converts a multiline JSON file (where each object is on a separate line)
    to a single-line JSON array.

    Args:
        input_filepath: Path to the input JSON file.
        output_filepath: Path to the output JSON file.
    """
    try:
        data = []
        with open(input_filepath, 'r') as infile:
            for line in infile:
                # Skip empty lines and whitespace-only lines
                if line.strip():
                    try:
                        json_object = json.loads(line)
                        data.append(json_object)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: '{line.strip()}'")
                        print(f"Error details: {e}")
                        return False #Return error if any line has error
        
        with open(output_filepath, 'w') as outfile:
            json.dump(data, outfile, indent=None, separators=(',', ':'))  # Use compact separators
        return True
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

# Example usage:
for i in range(19):
  # print(i)
  input_file = wiki40b_en_dir + str(f"validation{i}.json")  # Replace with your input file path
  output_file = wiki40b_save_dir + str(f"validation{i}.json") # Replace with your output file path
  if convert_multiline_json_to_single_line(input_file,output_file):
      print("Conversion successful!")
  else:
      print("Conversion failed!")

for i in range(1):
  # print(i)
  input_file = wiki40b_en_dir + str(f"validation{i}.json")  # Replace with your input file path
  output_file = wiki40b_save_dir + str(f"validation{i}.json") # Replace with your output file path
  if convert_multiline_json_to_single_line(input_file,output_file):
      print("Conversion successful!")
  else:
      print("Conversion failed!")