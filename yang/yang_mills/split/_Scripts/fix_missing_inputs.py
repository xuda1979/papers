import os
import re
import glob
import shutil

# Paths
root_dir = r"c:\Users\Lenovo\papers\yang\yang_mills\split"
unused_dir = os.path.join(root_dir, "_Unused_Appendices")
main_tex_path = os.path.join(root_dir, "main.tex")

def find_file_in_unused(filename):
    # exact match
    path = os.path.join(unused_dir, filename)
    if os.path.exists(path):
        return path
    return None

def main():
    try:
        with open(main_tex_path, 'r', encoding='latin-1') as f:
            content = f.read()

        # Regex to find \input{path/to/file}
        # Handling potential spaces or extensions
        includes = re.findall(r'\\input\{([^}]+)\}', content)

        for relative_path in includes:
            # Resolve full path
            full_path = os.path.normpath(os.path.join(root_dir, relative_path))
            
            if not os.path.exists(full_path):
                print(f"MISSING: {relative_path}")
                
                # Check if it's in a subdirectory in the input string
                # e.g. Ch3/file.tex -> filename is file.tex
                dirname, filename = os.path.split(relative_path)
                
                found_path = find_file_in_unused(filename)
                
                if found_path:
                    print(f"  FOUND in Unused: {found_path}")
                    # Construct destination directory
                    dest_dir = os.path.dirname(full_path)
                    
                    # Move the file
                    print(f"  MOVING to {dest_dir}")
                    if not os.path.exists(dest_dir):
                         os.makedirs(dest_dir)
                    
                    dest_path = os.path.join(dest_dir, filename)
                    shutil.move(found_path, dest_path)
                else:
                    print("  NOT FOUND in Unused Appendices.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
