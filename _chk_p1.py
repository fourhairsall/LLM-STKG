import re
p = r"D:\databuddy\专利写作\2026年7月\旅游推荐论文\code\multi_dataset_paper.tex"
out = r"D:\databuddy\专利写作\2026年7月\旅游推荐论文\code\_chk_out.txt"
s = open(p, encoding='utf-8').read()
lines = s.split('\n')
buf = []
m = re.search(r'\\title\{.*?\}', s, re.S)
buf.append('=== TITLE ===')
buf.append(m.group(0)[:300] if m else 'NONE')
buf.append('=== ROTAN / COMPETITIVE ===')
for i, ln in enumerate(lines, 1):
    if 'Rotan' in ln or ('competitive' in ln and 'not' not in ln.lower()):
        buf.append('L%d: %s' % (i, ln[:260]))
buf.append('=== MCNAMAR NOTATION ===')
for i, ln in enumerate(lines, 1):
    if 'McNemar' in ln and ('discordant' in ln or 'b' in ln or 'c' in ln):
        buf.append('L%d: %s' % (i, ln[:260]))
buf.append('=== FDR/BONFERRONI ===')
found = False
for i, ln in enumerate(lines, 1):
    if 'FDR' in ln or 'Bonferroni' in ln or 'BH ' in ln or 'multiple comparison' in ln.lower():
        buf.append('L%d: %s' % (i, ln[:200])); found = True
if not found:
    buf.append('NONE FOUND')
buf.append('=== CONTRIBUTIONS ===')
for i, ln in enumerate(lines, 1):
    if 'Contribution' in ln or 'contribut' in ln:
        buf.append('L%d: %s' % (i, ln[:200]))
buf.append('=== ABSTRACT p-values (L30-45) ===')
for i in range(29, 45):
    if i < len(lines):
        buf.append('L%d: %s' % (i+1, lines[i][:200]))
open(out, 'w', encoding='utf-8').write('\n'.join(buf))
print('WROTE', out)
