# #!/bin/python3

# import math
# import os
# import random
# import re
# import sys

# #
# # Complete the 'hitung' function below.
# #
# # The function is expected to return a STRING.
# # The function accepts STRING bilangan as parameter.
# #

# def hitung(bilangan):
#     # Write your code here
#     bin_bilangan = bin(bilangan).replace("0b", "")
#     arr = list(bin_bilangan)
#     j = -1
    
#     for i in range(len(bin_bilangan)):
#         if arr[i] == '1':
#             j+= 1
#             arr[i], arr[j] = arr[j], arr[i]
#     sort_bilangan = "".join(arr)
#     new_bilangan = "0b" + sort_bilangan
#     return (int(new_bilangan,2))


# if __name__ == '__main__':
#     N = int(input().strip())

#     for N_itr in range(N):
#         M = input()

#         X = hitung(M)

#         print(X + '\n')

# -------------------------N0 2 -------------------------
# #!/bin/python3

# import math
# import os
# import random
# import re
# import sys

# #
# # Complete the 'barisanBertopi' function below.
# #
# # The function is expected to return a LONG_INTEGER.
# # The function accepts following parameters:
# #  1. INTEGER n
# #  2. INTEGER_ARRAY H
# #  3. INTEGER_ARRAY C
# #

# def barisanBertopi(n, H, C):
#     # Write your code here
#     sum = 0
#     temp_sum = 0
#     for i in range(1,n):
#         temp_sum += C[i-1]
#         if H[i] >= H[i-1]:
#             sum += temp_sum
#     return sum

# if __name__ == '__main__':
#     n = int(input().strip())

#     H = list(map(int, input().rstrip().split()))

#     C = list(map(int, input().rstrip().split()))

#     result = barisanBertopi(n, H, C)

#     print(str(result) + '\n')

# ----------------------------------- No 3 -------------------------------
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'findTotalWays' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER N
#  2. STRING S
#

def findTotalWays(N, S):
    # Write your code here
    cnt = 0
    for i in range(1,N):
        if S[i] == '.':
            if S[i+1] != '_':
                cnt += 1
        
    return pow(2, cnt-1)

if __name__ == '__main__':
    N = 3

    S = 'o_.'

    result = findTotalWays(N, S)

    print(str(result) + '\n')