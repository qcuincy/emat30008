import sys, os
if not hasattr(sys.modules[__name__], '__file__'):
    emat_dir = os.path.dirname(os.path.abspath())
else:
    emat_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".."
        )
    )
sys.path.insert(0, emat_dir)

"""
Added docstrings for all functions
"""