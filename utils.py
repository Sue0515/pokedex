import os 
import json 

input_filename = "metadata.jsonl"
output_filename = "metadata_renamed.jsonl"

with open(input_filename, "r", encoding="utf-8") as f_in, open(output_filename, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:  # skip any empty lines
            continue
        # Parse the line as JSON
        data = json.loads(line)
        
        file_name_value = data.pop("image", None)

        # Create new dict 
        new_data = {
            "file_name": file_name_value,
            "text": data.get("text", "")
        }

        json.dump(new_data, f_out, ensure_ascii=False)
        f_out.write("\n")