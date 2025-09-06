echo "=== SECTION 1: BASICS ==="
echo "Hello, world!"

name="Student"
age=25
echo "My name is $name and I am $age years old."

current_date = $(date)
echo "Today is: $current_date"

echo

echo "=== SECTION 2: CONDITIONALS ==="
number=7
if [ $number -gt 10 ]; then
    echo "$number is greater than 10"
elif [ $number -eq 10 ]; then
    echo "$number is equal to 10"
else [$number -lt 10 ]; then
    echo "$number is less than ten"
fi

str="hello"
if [ "$str" = "hello" ]; then
    echo "The string is hello"
fi

echo

echo "===SECTION 3: LOOOPS==="

for i in {1..5}; do
    echo "For loop iteration: $i"
done

count=1
while [ $count -le 5 ]; do
    echo "While loop count: $count"
    ((count++))
done

num=1
until [ $num -gt 3 ]; do
    echo "Until loop number: $num"
    ((num++))
done

echo

echo "=== SECTION 4: ARRAYS ==="
fruits=("apple" "banana" "cherry")
echo "First fruit ${fruits[0]}"
echo "All fruits ${fruits[@]}"

for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

echo

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

echo "=== SECTION 6: FILES ==="

mkdir -p test_dir
echo "This is a sample file." > test_dir/sample.txt

echo "Contents of sample.txt:"
cat test_dir/sample.txt

echo "Appending a line..."
echo "Another line." >> test_dir/sample.txt
cat test_dir/sample.txt

echo

echo "=== SECTION 7: USER INPUT ==="
read -p "Enter your favorite color: " color
echo "You chose: $color"

echo
echo "=== SECTION 8: MATH ==="
a=5
b=3
echo "$a + $b = $((a + b))
echo "$a * $b = $((a * b))

rand = $((RANDOM % 10 + 1))
echo "Random number between 1 and 10: $rand"

echo

echo "=== SECTION 9: TEXT PROCESSING ==="
echo -e "one/ntwo/nthree" > list.txt

echo "File contents:"
cat list.txt

echo "Uppercase with tr:"
cat list.txt | tr '[:lower]' '[:upper:]'

echo "Line count with wc:"
wc -l list.txt

echo "Search with grep:"
grep "two" list.txt

echo

echo "=== SECTION 10: CASE ==="
read -p "Enter yes or no: " answer
case $answer in
    yes|y|Y) echo "You said YES";;
    no|n|N) echo "You said NO";;
    *) echo "Invalid response";;
esac

echo

echo "=== SECTION 11: SYSTEM INFO ==="
echo "current user: $USER"
echo "Home directory: $HOME"
echo "Shell: $SHELL"
echo "Uptime: $(uptime -p):"

echo

echo "=== SECTION 12: MULTIPLICATION TABLE ==="
for i in {1..5}; do
    for j in {1..5}; do
        printf "%4f $((i * j))"
    done
    echo
done

echo

echo "=== SECTION 13: ARGUMENTS ==="
echo "Number of arguments: $#"
echo "All arguments: $@"

if [ $# -ge 1 ]; then
    echo "First argument: $1"
fi

echo
echo "=== SECTION 14: EXIT CODES ==="
ls test_dir >/dev/null 2>&1
echo "Exit code of ls: $?"

echo

echo "=== SECTION 15: CLEANUP ==="
rm -rf test_dir list.txt
echo "Temporary files removed."

echo
echo "=== END OF SCRIPT ==="
