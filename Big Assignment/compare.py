
def read_output(filename):
    with open(filename, 'r') as file:
        output = [int(line) for line in file]
    return output

def read_levels(filename):
    with open(filename, 'r') as file:
        levels = [int(line) for line in file]
    return levels

def compare(output, levels):
    if output == levels:
        print("The output is correct!")
    else:
        print("The output is incorrect!")
        
def main():
    print("Comparing output.txt with bfs_levels.txt")
    output_filename = "output.txt"  # Change this to the filename of your output file
    levels_filename = "bfs_levels.txt"  # Change this to the filename of your levels file

    output = read_output(output_filename)
    levels = read_levels(levels_filename)
    
    compare(output, levels)
    

if __name__ == "__main__":
    main()