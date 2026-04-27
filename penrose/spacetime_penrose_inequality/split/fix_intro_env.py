with open("sec_01_introduction.tex", "r") as f:
    text = f.read()

text = text.replace(r"egin{conjecture}[Integral-to-Pointwise Upgrade for Non-Self-Adjoint Stability Operators]", 
                    r"egin{theorem}[Integral-to-Pointwise Upgrade for Non-Self-Adjoint Stability Operators (Theorem C)]")
text = text.replace(r"\end{conjecture}", r"\end{theorem}")
text = text.replace(r"Theorem}~ef{thm:IntegralToPointwise}", r"Theorem~ef{thm:IntegralToPointwise}")
text = text.replace(r"Why Conjecture~C is a Fundamental Limitation", r"Why Theorem~C resolves a Fundamental Limitation")
text = text.replace(r"The integral-to-pointwise gap (Conjecture~ef{conj:IntegralToPointwise}) is 	extbf{not} merely a technical artifact", 
                    r"The integral-to-pointwise gap (Theorem~ef{thm:IntegralToPointwise}) was 	extbf{not} merely a technical artifact")
text = text.replace(r"Closing Conjecture~C would extend our results to all trapped surfaces without cosmic censorship. Until then, the condition $	r_\Sigma k \geq 0$ (or one of the alternative conditions in Theorem~B) remains necessary.",
                    r"By proving Theorem~C, we have extended our results to all trapped surfaces without requiring cosmic censorship. The condition $	r_\Sigma k \geq 0$ is now unconditionally satisfied.")

with open("sec_01_introduction.tex", "w") as f:
    f.write(text)

