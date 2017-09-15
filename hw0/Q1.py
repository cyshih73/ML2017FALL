import sys

fp = open(sys.argv[1],'r')
line = fp.read()
templist = line.split("\n")
list = templist[0].split(" ")
a = 0
ans = []
for i in list:
	if i not in ans:
		ans.append(i)
		print(i, end=' ')
		print(a, end=' ')
		print(list.count(i))
		a += 1
	
