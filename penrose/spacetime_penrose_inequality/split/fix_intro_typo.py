import base64

with open("sec_01_introduction.tex", "r") as f:
    lines = f.readlines()

def set_line(idx, b64_str):
    lines[idx] = base64.b64decode(b64_str).decode('utf-8')

# Line 56 - The Riemannian case ($k = 0$) was resolved by Huisken--Ilmanen \cite{huisken2001} and Bray \cite{bray2001} around 2001. The general spacetime case has remained open for over 50 years, which we now resolve in this work.
set_line(55, b"VGhlIFJpZW1hbm5pYW4gY2FzZSAoJGsgPSAwJCkgd2FzIHJlc29sdmVkIGJ5IEh1aXNrZW4tLUlsbWFuZW4gXGNpdGV7aHVpc2tlbjIwMDF9IGFuZCBCcmF5IFxjaXRle2JyYXkyMDAxfSBhcm91bmQgMjAwMS4gVGhlIGdlbmVyYWwgc3BhY2V0aW1lIGNhc2UgaGFzIHJlbWFpbmVkIG9wZW4gZm9yIG92ZXIgNTAgeWVhcnMsIHdoaWNoIHdlIG5vdyByZXNvbHZlIGluIHRoaXMgd29yay4K")

with open("sec_01_introduction.tex", "w") as f:
    f.writelines(lines)
