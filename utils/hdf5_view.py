import h5py
import argparse
import sys

def print_attributes(obj, indent_level):
    if not obj.attrs:
        return
    
    prefix = ' ' * (indent_level + 2) + '└─ Attributes:'
    print(prefix)
    
    attr_prefix = ' ' * (indent_level + 4) + '├─'
    for key, value in obj.attrs.items():
        print(f"{attr_prefix} {key}: {value}")

def print_hdf5_structure(group, indent_level=0):
    indent = ' ' * indent_level
    
    for name, obj in group.items():
        if isinstance(obj, h5py.Group):
            print(f"{indent}├─ Group: {obj.name}")
            print_attributes(obj, indent_level)
            print_hdf5_structure(obj, indent_level + 2)
            
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}└─ Dataset: {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")
            print_attributes(obj, indent_level)
        
        else:
            print(f"{indent}└─ Other: {name} (Type: {type(obj)})")


def main():
    parser = argparse.ArgumentParser(
        description="A Python script for inspecting the internal structure of an HDF5 file."
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the HDF5 file to inspect."
    )
    
    args = parser.parse_args()
    filepath = args.filepath
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"--- HDF5 File Structure: {filepath} ---")
            print_hdf5_structure(f)
            print("----------------------------------------")
            
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Unable to read file '{filepath}'. It may not be a valid HDF5 file or it might be corrupted.", file=sys.stderr)
        print(f"H5py error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unknown error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

