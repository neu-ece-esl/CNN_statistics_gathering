import islpy as isl
# bset1 = isl.BasicSet("[n1, n2] -> {{[{}] : {} >= n1 and {} < n2}}".format('x', 'x', 'x'))
# bset2 = isl.BasicSet("{[x] : x >= 2 and x < 10}")
bset1 = isl.BasicSet()
bset1 = isl.BasicSet("{[x] : 0<=x<10 }")
bmap1 = isl.BasicMap("[n] -> {[x] -> [x'] : x'=2*x+n}").intersect_params(isl.BasicSet("[n] -> { : 0<=n<=3}"))
# bmap2 = isl.BasicMap("{[x] -> [ACC] : ACC = x}").intersect_domain(bset1)
# print(bmap1)
# print(bmap2)
# print(bmap2.reverse())
# print(bmap1.apply_range(bmap2.reverse()))
# res = bset1.lex_lt_union_set(bset1)
# print(bset1.union(bset2))
# bset1.card()
print(bset1)
res = bmap1.intersect_domain(bset1)
print(res)
