import random

with open("first_game_record.txt") as f:
    content = f.readlines()

content = [x.strip() for x in content] 
human_his = []
comp_his =[]
start = 0
close = 0
for i in range(len(content)):
    if start == 1 and content[i] == '':
        break
    if start == 1:
        human_his.append(content[i][0])
        comp_his.append(content[i][2])
    if content[i] == '':
        start = 1

game_hum = []

first_4_h = input("Enter the first 5 moves of you(ex. R P S R): ")

game_hum = first_4_h.split()

for i in range(len(game_hum)):
    if len(game_hum) > 5:
	    first_4_h = input("More than 5. Please enter again the first 4 moves of you(ex. R P S R): ")
	    game_hum = first_4_h.split()

    if game_hum[i] != 'R' and game_hum[i] != 'S' and game_hum[i] != 'P':
        first_4_h = input("Please enter again the first 5 moves of you(ex. R P S R): ")
        game_hum = first_4_h.split()


human_his = ''.join(human_his)
game_hum = ''.join(game_hum)

human_his += game_hum

while True:
	dic = {}

	for i in range(len(human_his)):
		if (i+3) == (len(human_his)-1):
			break
		key = human_his[i:i+4]
		if key not in dic:
			dic[key] = [human_his[i+4]]
		else:
			dic[key].append(human_his[i+4])

	human_pos = []
	if game_hum[-4:] in dic:
		human_pos = dic[game_hum[-4:]]

		S = 0
		P = 0
		R = 0
		result = ''
		for i in range(len(human_pos)):
			if human_pos[i] == 'S':
				S += 1
			elif human_pos[i] =='P':
				P += 1
			elif human_pos[i] == 'R':
				R += 1
		if max(S, P, R) == S:
			result = 'R'
		elif max(S, P, R) == P:
			result = 'S'
		elif max(S, P, R) == R:
			result = 'P'
		elif max(S, P, R) == 0:
			ran = random.randrange(1, 4)
			if ran == 1:
				result = 'R'
			if ran == 2:
				result = 'S'
			if ran == 3:
				result = 'P'
	else:
		ran = random.randrange(1, 4)
		if ran == 1:
			result = 'R'
		if ran == 2:
			result = 'S'
		if ran == 3:
			result = 'P'

	print("Please try: ", result)

	human_his += result

	a = input("input 0 to exit, 1 to continue: ")
	if a == '0':
		exit()
