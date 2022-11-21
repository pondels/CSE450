with open('text_files/hobbit.txt', 'r') as file:
    book = file.read()
    lines = book.split('\n')
    for line in lines:
        print(line)