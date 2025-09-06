#!/bin/bash
# practice.sh - A long Bash practice script with many examples
# Run with: bash practice.sh

########################################
# 1. Hello World and Variables
########################################
echo "=== SECTION 1: BASICS ==="
echo "Hello, World!"

# Variables
name="Student"
age=25
echo "My name is $name and I am $age years old."

# Command substitution
current_date=$(date)
echo "Today is: $current_date"

echo

########################################
# 2. Conditionals
########################################
echo "=== SECTION 2: CONDITIONALS ==="
number=7
if [ $number -gt 10 ]; then
    echo "$number is greater than 10"
elif [ $number -eq 10 ]; then
    echo "$number is equal to 10"
else
    echo "$number is less than 10"
fi

# String comparison
str="hello"
if [ "$str" = "hello" ]; then
    echo "The string is hello"
fi

echo

########################################
# 3. Loops
########################################
echo "=== SECTION 3: LOOPS ==="

# For loop
for i in {1..5}; do
    echo "For loop iteration: $i"
done

# While loop
count=1
while [ $count -le 5 ]; do
    echo "While loop count: $count"
    ((count++))
done

# Until loop
num=1
until [ $num -gt 3 ]; do
    echo "Until loop number: $num"
    ((num++))
done

echo

########################################
# 4. Arrays
########################################
echo "=== SECTION 4: ARRAYS ==="
fruits=("apple" "banana" "cherry")
echo "First fruit: ${fruits[0]}"
echo "All fruits: ${fruits[@]}"

for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

echo

########################################
# 5. Functions
########################################
echo "=== SECTION 5: FUNCTIONS ==="

say_hello() {
    echo "Hello from a function!"
}
say_hello

add_numbers() {
    local a=$1
    local b=$2
    echo "Sum: $((a + b))"
}
add_numbers 3 7

echo

########################################
# 6. File Operations
########################################
echo "=== SECTION 6: FILES ==="

mkdir -p test_dir
echo "This is a sample file." > test_dir/sample.txt

echo "Contents of sample.txt:"
cat test_dir/sample.txt

echo "Appending a line..."
echo "Another line." >> test_dir/sample.txt
cat test_dir/sample.txt

echo

########################################
# 7. User Input
########################################
echo "=== SECTION 7: USER INPUT ==="
read -p "Enter your favorite color: " color
echo "You chose: $color"

echo

########################################
# 8. Math and Random
########################################
echo "=== SECTION 8: MATH ==="
a=5
b=3
echo "$a + $b = $((a + b))"
echo "$a * $b = $((a * b))"

rand=$((RANDOM % 10 + 1))
echo "Random number between 1 and 10: $rand"

echo

########################################
# 9. Text Processing
########################################
echo "=== SECTION 9: TEXT PROCESSING ==="
echo -e "one\ntwo\nthree" > list.txt

echo "File contents:"
cat list.txt

echo "Uppercase with tr:"
cat list.txt | tr '[:lower:]' '[:upper:]'

echo "Line count with wc:"
wc -l list.txt

echo "Search with grep:"
grep "two" list.txt

echo

########################################
# 10. Case Statements
########################################
echo "=== SECTION 10: CASE ==="
read -p "Enter yes or no: " answer
case $answer in
    yes|y|Y) echo "You said YES";;
    no|n|N) echo "You said NO";;
    *) echo "Invalid response";;
esac

echo

########################################
# 11. System Info
########################################
echo "=== SECTION 11: SYSTEM INFO ==="
echo "Current user: $USER"
echo "Home directory: $HOME"
echo "Shell: $SHELL"
echo "Uptime: $(uptime -p)"

echo

########################################
# 12. Nested Loops (Multiplication Table)
########################################
echo "=== SECTION 12: MULTIPLICATION TABLE ==="
for i in {1..5}; do
    for j in {1..5}; do
        printf "%4d" $((i * j))
    done
    echo
done

echo

########################################
# 13. Script Arguments
########################################
echo "=== SECTION 13: ARGUMENTS ==="
echo "Number of arguments: $#"
echo "All arguments: $@"

if [ $# -ge 1 ]; then
    echo "First argument: $1"
fi

echo

########################################
# 14. Exit Codes
########################################
echo "=== SECTION 14: EXIT CODES ==="

ls test_dir >/dev/null 2>&1
echo "Exit code of ls: $?"

ls not_a_real_dir >/dev/null 2>&1
echo "Exit code of bad ls: $?"

echo

########################################
# 15. Cleanup
########################################
echo "=== SECTION 15: CLEANUP ==="
rm -rf test_dir list.txt
echo "Temporary files removed."

echo
echo "=== END OF SCRIPT ==="
