from random import randint

consonants = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','y','z']
vowels = ['a','e','i','o','u']

for i in range(250):
	l = randint(1,12)
	word = ''
	for j in range(l):
		if j%2 == 0:
			word += consonants[randint(0,len(consonants)-1)]
		else:
			word += vowels[randint(0,len(vowels)-1)]
	word += '#'
	print(word)