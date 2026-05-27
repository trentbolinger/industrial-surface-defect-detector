# A list of the 6 defect types from the NEU Surface Defect Dataset
defects = [
    "crazing",
    "inclusion",
    "patches",
    "pitted surface",
    "rolled-in scale",
    "scratches",
]

# Loop through the list and print each defect with its position number
for index, defect in enumerate(defects):
    print(f"{index + 1}: {defect}")

# A function that takes a defect name and returns a formatted detection message
def describe_defect(name):
    return f"Defect detected: {name}"

# Call the function on each defect and print the result
for defect in defects:
    print(describe_defect(defect))
