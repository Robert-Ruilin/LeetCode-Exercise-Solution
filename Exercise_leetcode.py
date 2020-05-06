"1.三个数求和等于0，去重"
def threeSum(nums):
    nums.sort()
    result = []
    n = len(nums)
    for i in range(n):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        head = i + 1
        end = n - 1
        while head < end:
            if nums[i] + nums[head] + nums[end] == 0:
                result.append([nums[i], nums[head], nums[end]])
                head += 1
                while head < end and nums[head] == nums[head-1]:
                    head += 1
                end -= 1
                while head < end and nums[end] == nums[end+1]:
                    end -= 1
            elif nums[i] + nums[head] + nums[end] < 0:
                head += 1
                while head < end and nums[head] == nums[head-1]:
                    head += 1
            else:
                end -= 1
                while head < end and nums[end] == nums[end+1]:
                    end -= 1
    return result
    
print(threeSum([1,0,-1,2,-2,4]))

"2. 两束求和，去重"
class Solution1:
    def twoSum(self, nums, target):
        sum = []
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums [i-1]:
                continue
            head = i + 1
            while head < n: 
                if nums[i] + nums[head] == target:
                    sum.append([nums[i],nums[head]])
                    head += 1
                    while head < n and nums[head] == nums[head - 1]:
                        head += 1
                else:
                    head += 1
                    while head < n and nums[head] == nums[head - 1]:
                        head += 1
        return sum

S = Solution1()
print(S.twoSum(nums=[4,1,2,5],target=6))

"2. 两束求和，唯一解"
class Solution2:
    def twoSum1(self, nums, target):
        solver = []
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            head = i + 1
            while head < n: 
                if nums[i] + nums[head] == target:
                    solver.append(i)
                    solver.append(head)
                    break
                else:
                    head += 1
                    while head < n and nums[head] == nums[head - 1]:
                        head += 1
            if solver == []:
                continue
            else:
                break
        return solver
    
    def twoSum2(self, nums, target):
        solver = {}
        for i, num in enumerate(nums):
            n = target - num
            if n not in solver:
                solver[num] = i
            else:
                return [solver[n],i]
        

S = Solution2()
print(S.twoSum1(nums=[2,1,2,5,5],target=6))
print(S.twoSum2(nums=[2,1,2,5,5],target=6))


"3. 数字反转"
class Solution3:
    def reverse1(self, x: int) -> int:
        x_s = str(abs(x))
        x_reverse = int(x_s[::-1])
        if x > 0:
            if  x_reverse > 2 ** 31 - 1:
                return 0
            else:
                return x_reverse
        else:
            if -1 * x_reverse < -2 ** 31:
                return 0
            else:
                return -1 * x_reverse
    
    def reverse2(self, x: int) -> int:
        if x == 0:
            return x
        else:
            x_s = str(abs(x))
            while x_s[-1] == str(0):
                x_s = x_s[:-1]
            num = []
            for i in x_s:
                num.append(i)
            n = len(num)
            if n % 2 == 0:
                for i in range (0, int(n/2)):
                    temp = num[i]
                    num[i] = num[-1-i]
                    num[-1-i] = temp
            else:
                for i in range (0, int((n-1)/2)):
                    temp = num[i]
                    num[i] = num[-1-i]
                    num[-1-i] = temp
            x_reverse = int("".join(num))
            if x > 0:
                if x_reverse > 2 ** 31 - 1:
                    return 0
                else:
                    return x_reverse
            else:
                if -1 * x_reverse < -2 ** 31:
                    return 0
                else:
                    return -1 * x_reverse

S = Solution3()
print(S.reverse1(-2424))
print(S.reverse2(-2424))

"4. str - int"
class Solution4:
    def myAtoi(self, str: str) -> int:
        sign = ['-','+']
        num = ['0','1','2','3','4','5','6','7','8','9']
        new = []
        if str == '' or str in sign :
            return 0
        else:
            while str[0] == ' ':
                str = str[1:]
                if str == '':
                    return 0
        if str[0] in sign:
            i = 1
            while str[i] in num:
                new.append(str[i])
                if i == len(str) - 1:
                    break
                else:
                    i += 1
            if new == []:
                return 0
            elif str[0] == sign[0]:
                return max(-1 * int(''.join(new)), -2**31)
            elif str[0] == sign[1]:
                return min(int(''.join(new)), 2**31 - 1)
        else:
            i = 0
            while str[i] in num:
                new.append(str[i])
                if i == len(str) - 1:
                    break
                else:
                    i += 1
            if new == []:
                return 0
            else:
                return min(int(''.join(new)), 2**31 - 1)

S = Solution4()
print(S.myAtoi('  '))

"5. 数之后比自己小的数字数量"
class Solution5:
    def countSmaller(self, nums):
        counts = [0 for i in range(len(nums))]
        for i in range(len(nums)):
            for j in range(i,len(nums)):
                if nums[i] > nums[j]:
                    counts[i] = counts[i] + 1
        return counts

S = Solution5()
print(S.countSmaller([1,2,56,3,1,6]))

"6. 杨辉三角"
class Solution6:
    def generate(self, numRows):
        triangle = []
        if numRows == 0:
            return triangle
        elif numRows == 1:
            triangle.append([1])
            return triangle
        else:
            triangle = [[1]]
            for i in range(1, numRows):
                row = [1]
                for j in range(1,i):
                    row.append(triangle[i-1][j-1]+triangle[i-1][j])
                row.append(1)
                triangle.append(row)
            return triangle
    
    def getRow(self, rowIndex):
        if rowIndex == 0:
            return [[1]]
        else:
            triangle = [[1]]
            for i in range(0,rowIndex):
                row = [1]
                for j in range(0,i):
                    row.append(triangle[i][j] + triangle[i][j+1])
                row.append(1)
                triangle.append(row)
            return triangle[rowIndex]

S = Solution6()
print(S.generate(10))
print(S.getRow(1))

"7. Palindrome Number"
class Solution7:
    def isPalindrome(self, x: int) -> bool:
        num = str(x)
        return num == num[::-1]
        
S = Solution7()
print(S.isPalindrome(-121))

"8. Longest Common Prefix"
class Solution8:
    def longestCommonPrefix(self, strs):
        indice = [len(strs[i]) for i in range(len(strs))]
        prefix = []
        for i in range(min(indice)):
            count = 0
            for j in range(len(strs)-1):
                if strs[j][i] == strs[j+1][i]:
                    count += 1
            if count == len(strs) - 1:
                prefix.append(strs[0][i])
            else:
                break
        return ''.join(prefix)
    
S = Solution8()
print(S.longestCommonPrefix(["flower","flow","flight"]))

"9. Valid Parentheses"
class Solution9:
    def isValid1(self, s: str) -> bool:
        bracket = ['()','{}','[]']
        if s == '': return True
        if len(s) % 2 == 0:
            if all(s[2*i] + s[2*i+1] in bracket for i in range(len(s)//2)):
                return True
            else:
                return False
        else:
            return False
    
    def isValid2(self, s):
        while "()" in s or "{}" in s or '[]' in s:
            s = s.replace("()", "").replace('{}', "").replace('[]', "")
        return s == ''
    
    def isValid3(self,s):  #stack algorithm
        bracket_map = {"(": ")", "[": "]",  "{": "}"}
        open_par = set(["(", "[", "{"])
        stack = []
        for i in s:
            if i in open_par:
                stack.append(i)
            elif stack and i == bracket_map[stack[-1]]:
                stack.pop()
            else:
                 return False
        return stack == []

S = Solution9()
print(S.isValid1("()[]{}[][]"))
print(S.isValid2("{()[]{}[]}"))
print(S.isValid3("{()[]{}[]}"))

"10. Search insert position"
class Solution10:
    def searchInsert(self, nums, target: int) -> int:
        if nums == []:
            return 0
        if max(nums) < target:
            return len(nums)
        for i in range(len(nums)):
            if nums[i] >= target:
                return i

S = Solution10()
print(S.searchInsert([1,2,3,5,6,7],10))

"11. 数独 T/F"
class Solution11:
    def isValidSudoku(self, board) -> bool:
# row recursion
        for i in range(len(board)):
            if len(''.join(board[i]).replace('.','')) != len(set(''.join(board[i]).replace('.',''))):
                return False
# column recursion
            column = []
            for j in range(len(board[0])):
                if board[j][i] != '.':
                    column.append(board[j][i])
            if len(column) != len(set(column)):
                return False
# sub-box recursion
        index = [[0,1,2],[3,4,5],[6,7,8]]
        for a in range(3):
            for b in range(3):
                sub = []
                for i in index[a]:
                    for j in index[b]:
                        if board[i][j] != '.':
                            sub.append(board[i][j])
                if len(sub) != len(set(sub)):
                    return False
        return True
        
        

S = Solution11()
print(S.isValidSudoku([
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]))
    
"12. Max subarray"
class Solution12:
    def maxSubArray(self, nums) -> int:
        for i in range(1,len(nums)):
            nums[i] = max(nums[i-1]+nums[i],nums[i])
        return max(nums)

S = Solution12()
print(S.maxSubArray([2,-3,6]))

"13. Climbing stairs"
class Solution13:
    def permutation1(self, n: int, m: int) -> int:
        if m == 0: return 1
        num = n
        i = 1
        while i <= m - 1:
            num *= n - i
            i += 1
        product = 1
        for i in range(1,m+1):
            product *= i
        num = num // product
        return num
        
    def permutation2(self, n: int, m: int) -> int:
        if m == 0 or m == n:
            return 1
        else:
            return Solution13.permutation2(self, n-1,m) + Solution13.permutation2(self, n-1,m-1)
    
    def climbStairs(self, n: int) -> int:
        solver = []
        for a in range(0,n+1):
            for b in range(0,n//2+1):
                if a + 2 * b == n:
                    solver.append([a,b])
                    break
        step = 0
        for i in range(len(solver)):
            step += Solution13.permutation2(self, solver[i][0]+solver[i][1], solver[i][1])
        return step

S = Solution13()
print(S.climbStairs(4))

"14. Longest Substring Without Repeating Characters"
class Solution14:
    def lengthOfLongestSubstring(self, s: str) -> int:
        sub_s = ''
        length = 0
        max_length = 0
        for i in range(len(s)):
            if s[i] not in sub_s:
                sub_s += s[i]
                length += 1
            else:
                sub_s = sub_s[sub_s.find(s[i])+1:]
                sub_s  += s[i]
                length = len(sub_s)
            max_length = max(length, max_length)
        return max_length

S = Solution14()
print(S.lengthOfLongestSubstring('abcdeagdcikdlo'))

"15. 3Sum Clostest"
class Solution15:
    def threeSumClosest(self, nums, target: int) -> int:
        nums.sort()
        mindiff = sys.maxsize
        result = 0
        for i in range(len(nums)):
            #判断相邻两数是否相同，如果相同，直接进入下一次循环
            if i > 0 and nums[i] == nums[i-1]:
                continue
            head = i + 1
            end = len(nums) - 1
            while head < end:
                #判断三个数之和与目标值的距离是否小于mindiff值，如果是，替换mindiff
                if abs(nums[i] + nums[head] + nums[end] - target) < mindiff:
                    mindiff = abs(nums[i] + nums[head] + nums[end] - target)
                    result = nums[i] + nums[head] + nums[end]
                #判断三数之和与目标值的大小关系，如果三数之和大，尾部index减一，如果目标值大，头部index加一，如果一样大，输出结果
                if nums[i] + nums[head] + nums[end] == target:
                    return nums[i] + nums[head] + nums[end]
                elif nums[i] + nums[head] + nums[end] < target:
                    head += 1
                else:
                    end -= 1
        return result

S = Solution15()
print(S.threeSumClosest([-1, 2, 1, -4],3))

"16. k-sum problem"
class Solution16:
    def fourSum(self, nums, target):
        def helper(nums,target,ksum,begin):
            if begin >= len(nums): #base condition
                return []
            if ksum == 2:  
			 #two pointer approach as we do in 2sum problem
                res = []
                i = begin
                j = len(nums) - 1
                while i < j:
                    if nums[i] + nums[j] == target:
                        res.append([nums[i], nums[j]])
                        while i < j and nums[i] == nums[i + 1]:
                            i += 1
                        while i < j and nums[j] == nums[j - 1]:
                            j -= 1
                        i += 1
                        j -= 1
                    elif nums[i] + nums[j] > target:
                        j -= 1
                    else:
                        i += 1
                return res
            res = [] 
            for i in range(begin, len(nums) - ksum + 1):
                if i > begin and nums[i] == nums[i - 1] or nums[i] + nums[-1] * (ksum - 1) < target:
                    continue
					# the first condition is to remove Duplicates 
					# the second Condition is that if we consider element nums[i] whether we get required target or not
					#ie if we consider nums[i] which is smallest in current ksum size windown and ksum-1 times of largest number in the nums and still we get sum < target the we do not need to consider nums[i]
                if nums[i] + nums[i + 1] * (ksum - 1) > target:
                    break
					#Same as above   if nums[i] + nums[i + 1] * (ksum - 1) > target it means that remaining elements after i algo give result > target hence we exit from the loop
                r = helper(nums, target - nums[i], ksum - 1, i + 1)
                for elm in r:
                    elm.insert(0,nums[i])
					#We insert current nums[i] to the result of ksum-1 result
                for elm in r:
                    res.append(elm)
					#we store result of size ksum 
            return res

        nums.sort()
        ksum = 3
		#ksum = 4 As we are finding 4sum here
		# For generailized  case 
		#if ksum <1: return []
		#if ksum ==1:
			# return  [target]*nums.count(target)
        return helper(nums, target, ksum, 0)

S = Solution16()
print(S.fourSum([-1, 2, 1, -4,4,-2],3))

"17.Factorial Trailing Zeroes"
class Solution17:
    def trailingZeroes(self, n: int) -> int:
        product = 1
        for i in range(n,0,-1):
            product *= i
        product_rev = int(str(product)[::-1])
        return len(str(product)) - len(str(product_rev))

S = Solution17()
print(S.trailingZeroes(10))

"18.majority elements"
class Solution18:
    def majorityElement1(self, nums) -> int:
        dict_major = {}
        for i in nums:
            if i in dict_major.keys():
                dict_major[i] += 1
            else:
                dict_major[i] = 1
        for i in dict_major.keys():
            if dict_major[i] > len(nums) // 2:
                return i
    
    def majorityElement2(self, nums) -> int:
        nums.sort()
        mid = nums[len(nums) // 2]
        count = 0
        for i in nums:
            if i == mid:
                count += 1
        if count > len(nums) // 2:
            return mid

S = Solution18()
print(S.majorityElement1([2,1,1]))
print(S.majorityElement2([2,1,1]))
