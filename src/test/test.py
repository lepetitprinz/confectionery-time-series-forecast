def check_angle(numbers):
    numbers = sorted(numbers)
    if numbers[0] ** 2 + numbers[1] ** 2 == numbers[2] ** 2:
        flag = 'right'
    else:
        flag = 'wrong'

    return flag

while True:
    numbers = list(map(int, input().split()))

    if numbers == [0, 0, 0]:
        break
    else:
        flag = check_angle(numbers)
        print(flag)



