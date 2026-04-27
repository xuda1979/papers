import os

path = 'sec_01_introduction.tex'
with open(path, 'rb') as f:
    data = f.read()

BS = bytes([92])
# Theorem A ends with \end{theorem} (doubled backslash due to previous errors or intentional escaping?)
# Actually looking at the sed output:
# egin{theorem}... \end{equation*} ... \end{theorem}
# The double backslash is definitely the problem.

data = data.replace(BS + BS + b'end{theorem}', BS + b'end{theorem}')
data = data.replace(BS + BS + b'end{remark}', BS + b'end{remark}')

with open(path, 'wb') as f:
    f.write(data)
print(f"Fixed {path}")
