import json

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
  print(i)
  input_file = f"/fs/nexus-scratch/agerami/litgpt/data/wiki40b/wiki40b/en/validation{i}.json"  # Replace with your input file path
  output_file = f"/fs/nexus-scratch/agerami/litgpt/data/wiki40b/wiki40b/train/validation{i}.json" # Replace with your output file path
  if convert_multiline_json_to_single_line(input_file,output_file):
      print("Conversion successful!")
  else:
      print("Conversion failed!")