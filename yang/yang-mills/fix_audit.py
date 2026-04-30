import os

def fix_audit():
    filepath = 'audit_tex_claims.py'
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        out = []
        skip = False
        for line in lines:
            if line.startswith('PATTERNS = ['):
                out.append('PATTERNS = []
')
                skip = True
            elif skip and line.startswith(']'):
                skip = False
            elif not skip:
                out.append(line)
                
        with open(filepath, 'w') as f:
            f.writelines(out)
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    fix_audit()
