import math


score = 60

if score >= 80:
	grade = '优秀'
elif score >= 60:
	grade = '及格'
else:
	grade = '不及格'

print(grade)

score = 60

grade = '优秀' if score >= 80 else '及格' if score >= 60 else '不及格'

print(grade)
