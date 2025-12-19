"""
Test sanitization functions for data leakage prevention.
"""

import sys
sys.path.insert(0, '.')

from graph_contrastive.preprocess_graph import clean_latex, clean_html

def test_latex_sanitization():
    print("=" * 60)
    print("Testing LaTeX Sanitization")
    print("=" * 60)

    test_cases = [
        # (input, description)
        (r"By Lemma \ref{lem3.1} and Lemma \ref{lem3.2}, we have", "\\ref"),
        (r"From \eqref{meq}, for any $t\ge0$", "\\eqref"),
        (r"By \autoref{prop:1T}, the result follows", "\\autoref"),
        (r"Using \Cref{ex:st-triangle-E} and \cref{prop:basic}", "\\Cref and \\cref"),
        (r"See \hyperref[thm:main]{the main theorem}", "\\hyperref"),
        (r"\begin{lemma}\label{lem3.5} Let $x \in M$", "\\label"),
        (r"as shown in \cite{berthelotI} and \citet{ZLTW2018}", "\\cite variants"),
        (r"By \pageref{sec:intro}, we see that", "\\pageref"),
    ]

    all_passed = True
    for text, desc in test_cases:
        cleaned = clean_latex(text)
        # Check that the leaky pattern is removed
        has_leak = any(p in cleaned for p in [r'\ref', r'\eqref', r'\autoref', r'\cref',
                                               r'\Cref', r'\hyperref', r'\label', r'\cite',
                                               r'\pageref'])
        status = "PASS" if not has_leak else "FAIL"
        if has_leak:
            all_passed = False
        print(f"\n[{status}] {desc}")
        print(f"  Input:  {text}")
        print(f"  Output: {cleaned}")

    return all_passed


def test_html_sanitization():
    print("\n" + "=" * 60)
    print("Testing HTML/URL Sanitization")
    print("=" * 60)

    test_cases = [
        (r"See https://doi.org/10.1016/j.apnum.2016.11.004 for details", "DOI URL"),
        (r"Check http://en.wikipedia.org/wiki/65536_%28number%29", "Wikipedia URL"),
        (r"Visit www.mathoverflow.net for more", "www URL"),
        (r"<a href='link'>text</a> and more <b>bold</b>", "HTML tags"),
    ]

    all_passed = True
    for text, desc in test_cases:
        cleaned = clean_html(text)
        # Check that URLs and HTML are removed
        has_leak = any(p in cleaned.lower() for p in ['http://', 'https://', 'www.', '<a', '<b'])
        status = "PASS" if not has_leak else "FAIL"
        if has_leak:
            all_passed = False
        print(f"\n[{status}] {desc}")
        print(f"  Input:  {text}")
        print(f"  Output: {cleaned}")

    return all_passed


def test_real_samples():
    """Test on actual samples from the inspection."""
    print("\n" + "=" * 60)
    print("Testing Real Samples from Dataset")
    print("=" * 60)

    arxiv_samples = [
        r"Let $x \in M$. By definition, using Lemma \ref{lem3.1} and Lemma \ref{lem3.2}, we have $d(x, A) \leq |x-y|$",
        r"Since $C$ is $2$--realizable. By Lemma \ref{4realize}, the diagonal $\Delta$ is realizable",
        r"From \eqref{meq}, for any $t\ge0$ and $j=1,2$, we have $\rho_{t}^{j}$",
        r"The result follows from \autoref{prop:1T}, where we prove that $\mathcal{B}_n$ is the unit ball",
        r"Since the category has descent, by \cite{berthelotI}, we may assume $X$ is affine",
    ]

    se_samples = [
        r"See https://doi.org/10.1016/j.apnum.2016.11.004 for the bounds on coefficients",
        r"Why is $2^{16}=65536$ special? See http://en.wikipedia.org/wiki/65536_%28number%29",
    ]

    print("\nArXiv samples:")
    for i, text in enumerate(arxiv_samples, 1):
        cleaned = clean_latex(text)
        print(f"\n  Sample {i}:")
        print(f"    Before: {text[:80]}...")
        print(f"    After:  {cleaned[:80]}...")

    print("\nStackExchange samples:")
    for i, text in enumerate(se_samples, 1):
        cleaned = clean_html(text)
        print(f"\n  Sample {i}:")
        print(f"    Before: {text[:80]}...")
        print(f"    After:  {cleaned[:80]}...")


if __name__ == "__main__":
    latex_ok = test_latex_sanitization()
    html_ok = test_html_sanitization()
    test_real_samples()

    print("\n" + "=" * 60)
    if latex_ok and html_ok:
        print("All sanitization tests PASSED!")
    else:
        print("Some tests FAILED - review output above")
    print("=" * 60)
