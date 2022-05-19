def test(number, k):
    length = len(number)
    n = length - k

    result = []
    start_idx = 0
    end_idx = length - n
    for i in range(n):
        max_num = max(number[start_idx:end_idx+1])
        result.append(max_num)
        start_idx += number[start_idx:].index(max_num) + 1
        end_idx += 1

    return result

number = "4177252841"
k = 4

result = test(number, 4)
print(result)