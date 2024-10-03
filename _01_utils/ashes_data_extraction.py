from time import perf_counter
import re
import numpy as np
import pandas as pd
res_file_path = r"C:\Users\davis\00_data\00_Documents\01_Master_studies\46300 - WTT and Aerodynamics\01_Assigments\01_ashes\03_V_20\Sensor Rotor.txt"


#%%My first approach

def parse_sensor_data_first_try(file_path):
    with open(file_path, "r") as f:
        res_file = f.read()
    
    #Get column names
    # Regex pattern: Search for "Column" followed by a line of numbers and tabs,
    # followed by any a line with arbitrary characters, followed by one or more
    # hyphens followed by a tab.
    match = re.search(r"(?:Column\s*\n(?:\d+\t)+\n)(.+)(?:\n-+\t)+?", res_file) 
    col_names = match.group(1).split("\t")
    unit_dict = {re.sub(r"\s+\[.*\]", "", s) 
                 : re.search(r"(?:\s+\[)(.*)(?:\])", s).group(1) 
                 for s in col_names}
    col_names = [re.sub(r"\s+\[.*\]", "", s) for s in col_names]
    
    #Extract values
    #Regex pattern: Search for arbitrary characters (including newline), followed 
    # by a line of hyphens (both greedy quantifiers as non-capturing groups)
    # followed by arbitrary characters (including newline) follwed by a final newline
    match = re.search(r"(?:[\s\S]+)(?:\n(?:-*\t-+)+\n)([\s\S]+)(?:\n+)", res_file)
    
    data_str = match.group(1).split("\n")
    
    data = np.zeros((len(data_str), len(col_names)))
    for i, line in enumerate(data_str):
        try: 
            data[i,:] = np.array(data_str[i].split("\t")).astype(float)
        except Exception as e:
            print(data_str[i])
            
    data_df = pd.DataFrame(columns = col_names, data=data)
    return data_df


#%% Adjusted version from ChatGPT
import pandas as pd
import numpy as np
import re

def parse_sensor_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the row with column headers (sensor names)
    for i, line in enumerate(lines):
        if line.startswith('Time'):
            header_index = i
            break

    # Extract column headers (sensor names without units)
    raw_headers = lines[header_index].strip().split('\t')
    unit_dict = {re.sub(r"\s+\[.*\]", "", s).strip() 
                 : re.search(r"(?:\s+\[)(.*)(?:\])", s).group(1) 
                 for s in raw_headers}
    headers = unit_dict.keys()
    
    # Skip until the line that begins with the first data entry
    data_start_index = header_index + 7
    
    # Extract time series data
    data = np.zeros((len(lines)-data_start_index, len(headers)))
    for i, line in enumerate(lines[data_start_index:]):
        if line.strip():  # Skip empty lines
            data [i,:] = [float(value) for value in line.strip().split('\t')]

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    return df



#%% Original ChatGPT

def parse_sensor_data_GPT(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the row with column headers (sensor names)
    for i, line in enumerate(lines):
        if line.startswith('Time'):
            header_index = i
            break

    # Extract column headers (sensor names without units)
    raw_headers = lines[header_index].strip().split('\t')
    unit_dict = {re.sub(r"\s+\[.*\]", "", s).strip() 
                 : re.search(r"(?:\s+\[)(.*)(?:\])", s).group(1) 
                 for s in raw_headers}
    headers = unit_dict.keys()
    
    # Skip until the line that begins with the first data entry
    data_start_index = header_index + 7
    
    # Extract time series data
    data = []
    for line in lines[data_start_index:]:
        if line.strip():  # Skip empty lines
            data.append([float(value) for value in line.strip().split('\t')])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    return df

#%% Testing

t_gpt = []
t_me = []

for i in range(20):
    start = perf_counter()
    _ = parse_sensor_data(res_file_path)
    end = perf_counter()
    
    t_me.append(end-start)
    
    start = perf_counter()
    _ = parse_sensor_data_GPT(res_file_path)
    end = perf_counter()
    
    t_gpt.append(end-start)
    
print (f"ChatGPT: Mean = {np.round(np.mean(t_gpt),3)},"
       + f"Min = {np.round(np.min(t_gpt),3)}"
       + f"Max = {np.round(np.max(t_gpt),3)}")
print (f"Me: Mean = {np.round(np.mean(t_me),3)},"
       + f"Min = {np.round(np.min(t_me),3)}"
       + f"Max = {np.round(np.max(t_me),3)}")
