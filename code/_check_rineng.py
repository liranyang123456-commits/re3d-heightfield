# -*- coding: utf-8 -*-
import re
t = open(r'E:\MIS_TMI_Re_3D\re3d_cmpb_results\paper\paper_cmig.tex', encoding='utf-8').read()

m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
abs_text = m.group(1)
plain = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', abs_text)
plain = re.sub(r'[{}\\$]', ' ', plain)
print('abstract words (approx):', len(plain.split()))

mk = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', t, re.S)
kws = [k.strip() for k in re.split(r'\\sep|,', mk.group(1)) if k.strip()]
print('keywords:', len(kws), kws)

mj = re.search(r'\\journal\{[^}]*\}', t)
print('journal:', mj.group(0))

# CRediT check
print('has CRediT:', 'CRediT' in t or 'Conceptualization' in t)
# find author contributions paragraph
mc = re.search(r'Authors?. contributions.*?\n(.*?)\n\\', t, re.S)
print('contrib snippet:', (mc.group(1)[:200] if mc else 'NOT FOUND'))
