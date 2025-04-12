import pandas as pd
import glob
import os

# Patterns for capturing layer statistics
import re
layer_pattern = re.compile(r'(\S+)\s+: rmse ([\d\.eE+-]+), maxerr ([\d\.eE+-]+), 95pct<([\d\.eE+-]+), median<([\d\.eE+-]+)')
histogram_bin_pattern = re.compile(r'\[([\d\.]+), ([\d\.inf]+)\):\s+(\d+)')

def parse_filename(filename):
    basename = os.path.basename(filename)
    parts = basename.replace('.out', '').split('_')
    
    model = parts[0]
    idx = 1

    compression_type = None
    param_min = None
    param_max = None

    if parts[idx] == 'from':
        compression_type = parts[idx + 1].split('-')[1]
        param_min, param_max = parts[idx + 2].split('-')
        idx += 3

    quantization = parts[idx]
    idx += 1

    imat = parts[idx]
    idx += 1

    dim = None
    if idx < len(parts) and parts[idx].startswith('dim'):
        dim = parts[idx].split('dim')[1]

    return {
        'model': model,
        'compression_type': compression_type,
        'param_min': param_min,
        'param_max': param_max,
        'quantization': quantization,
        'imat': imat,
        'dim': dim
    }

def parse_model(filename:str):
    """
    Meta-Llama-3.1-8B-F16_Q4_1_no_imat
    Meta-Llama-3.1-8B-F16_from_ZFP-acc_0.01-0.13_wi_imat_dim_1
    """ 
    basename = os.path.basename(filename)
    modelname = basename.replace('.out', '')
    
    model_name = modelname.strip().removeprefix("Meta-Llama-")
    
    llama_version=model_name.split("-",1)[0] # 3
    num_parameter=model_name.split("-",2)[1] # 8B
    processing_type=model_name.split("-",2)[2].split("_",1)[0] # F16
    
    if "ZFP" in model_name:
        model_wo_prefix = model_name.split("-",2)[2].split("-",1)[1] # rate ...
        dim = model_wo_prefix[-(len("dim_X")):].split("_")[1] # dime
        type = model_wo_prefix.split("_",1)[0]
        threshold_low = model_wo_prefix.split("_",2)[1].split("-")[0]
        threshold_high = model_wo_prefix.split("_",2)[1].split("-")[1]
        imat = "_".join(model_wo_prefix.split("_")[2:4])
    
    else:
        model_wo_prefix = model_name.split("-",2)[2].removeprefix(processing_type)
        imat = "_".join(model_wo_prefix.split("_")[-2:])
        type="quantization"
        threshold_low = model_wo_prefix.removesuffix(imat).strip("_")
        threshold_high="<empty>"
        dim = 0
    print(f"{model_name=} {llama_version=} {num_parameter=} {processing_type=} {type=} {dim=} {threshold_low=} {threshold_high=} {imat=}")
    # return {
    #     'model': model,
    #     'compression_type': compression_type,
    #     'param_min': param_min,
    #     'param_max': param_max,
    #     'quantization': quantization,
    #     'imat': imat,
    #     'dim': dim
    # }
    return {
            "model_name":model_name,
            "llama_version":llama_version,
            "num_parameter":num_parameter,
            "processing_type":processing_type,
            "type":type,
            "dim":dim,
            "threshold_low":threshold_low,
            "threshold_high":threshold_high,
            "imat":imat
    }

def parse_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    setup = parse_model(filename)

    layers_data = []
    current_layer = None

    for line in lines:
        layer_match = layer_pattern.match(line)
        if layer_match:
            current_layer = {
                'layer': layer_match.group(1),
                'rmse': float(layer_match.group(2)),
                'maxerr': float(layer_match.group(3)),
                '95pct': float(layer_match.group(4)),
                'median': float(layer_match.group(5)),
                'histogram': []
            }
            layers_data.append(current_layer)
        elif current_layer:
            histogram_match = histogram_bin_pattern.match(line.strip())
            if histogram_match:
                bin_data = {
                    'bin_start': float(histogram_match.group(1)),
                    'bin_end': float(histogram_match.group(2)) if histogram_match.group(2) != 'inf' else float('inf'),
                    'count': int(histogram_match.group(3))
                }
                current_layer['histogram'].append(bin_data)

    return {**setup, 'layers': layers_data}

def process_all_files(pattern="*.out"):
    files = glob.glob(pattern)
    results = []
    for filename in files:
        try:
            data = parse_file(filename)
            data['filename'] = filename
            results.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results

if __name__ == "__main__":
# Example usage:
    search_dir="Meta-Llama-3.1-8B/result_tensor_comparison/"
    # Example usage:
    data = process_all_files(f"{search_dir}*.out")

    # Convert to DataFrame for summary
    df = pd.json_normalize(data, 'layers', ['llama_version', 'num_parameter', 'processing_type', 'type', 'dim', 'threshold_low', 'threshold_high', 'imat','model_name'], errors='ignore')
    print(df)

    # Save full summary to CSV
    df.to_csv("summary.csv", index=False)

    # Create second DataFrame without histogram column
    df_no_hist = df.drop(columns=['histogram'])
    df_no_hist = df_no_hist[df_no_hist['layer'] == 'global']
    print(df_no_hist)

    # Save second summary to CSV
    df_no_hist.to_csv("summary_no_histogram.csv", index=False)

