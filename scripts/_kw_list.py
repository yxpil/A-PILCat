# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, '.')
from pack_sr import collect_items, keyword_of, fix_keyword
kws = []
for name in collect_items():
    kws.append(fix_keyword(keyword_of(name)))
open('_kw_list.txt', 'w', encoding='utf-8').write('\n'.join(kws))
print(len(kws))
