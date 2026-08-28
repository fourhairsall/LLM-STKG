import sys
steps = [
    ("torch", "import torch; print('torch', torch.__version__)"),
    ("numpy", "import numpy; print('numpy', numpy.__version__)"),
    ("scipy", "import scipy; print('scipy', scipy.__version__)"),
    ("torch_geometric", "import torch_geometric; print('tg', torch_geometric.__version__)"),
    ("networkx", "import networkx; print('nx', networkx.__version__)"),
    ("sklearn", "import sklearn; print('sklearn', sklearn.__version__)"),
]
for name, code in steps:
    try:
        exec(code)
        sys.stdout.flush()
    except Exception as e:
        print("IMPORT_FAIL %s: %r" % (name, e))
        sys.stdout.flush()
        break
print("IMPORT_DONE")
sys.stdout.flush()
