from utils import run_cpp_file

# Load the C++ code
cpp = run_cpp_file("animated-umbrella/cpps/test.cpp")

# Use the functions
numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
total = cpp.sum_vector(numbers)
avg = cpp.average(numbers)

print(f"Sum: {total}, Average: {avg}")

# Create an instance of the Statistics class
stats = cpp.Statistics(numbers)
print(f"Mean: {stats.mean()}")
print(f"Max: {stats.max()}")
print(f"Min: {stats.min()}")

# Add a value and print data
stats.add_value(10.0)
stats.print_data()

