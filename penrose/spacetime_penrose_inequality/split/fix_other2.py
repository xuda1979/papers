import base64

def set_line(filename, idx, b64_str):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines[idx] = base64.b64decode(b64_str).decode('utf-8')
    with open(filename, "w") as f:
        f.writelines(lines)

# sec_02_the_penrose_conjecture.tex, line 45 (idx 44)
set_line("sec_02_the_penrose_conjecture.tex", 44, b"ICAgIFxpdGVtWyhCKV0gXHRleHRiZntDb21wYWN0bmVzczp9IE9uZSBvZiBjb25kaXRpb25zIChDMSktLShDMykgb2YgVGhlb3JlbX5ccmVme3RobTpNYXhBcmVhVHJhcHBlZH0gaG9sZHMsIGFsb25nIHdpdGggdGhlIG5vdyB1bmNvbmRpdGlvbmFsbHkgcHJvdmVuIFRoZW9yZW0gQyAoVGhlb3JlbX5ccmVme3RobTpJbnRlZ3JhbFRvUG9pbnR3aXNlfSksIG9yCg==")

