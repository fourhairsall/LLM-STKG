import sys, torch
with open("D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_smoke.out","w") as f:
    f.write("OK torch=%s cuda=%s\n" % (torch.__version__, torch.cuda.is_available()))
print("SMOKE_DONE")
