import csv
import sys

new_lines = []

with open(sys.argv[1], "r") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        reddit = row[0][1:-1].replace("'", "").strip().split(',')
        reddit = list(map(int, reddit))
        embed_text = row[1][1:-1].replace("'", "").strip().split(',')
        embed_text = list(map(int, embed_text))
        line = reddit + embed_text
        new_lines.append(line)

with open(sys.argv[2], "w") as csvfile:
    filewriter = csv.writer(csvfile)
    for line in new_lines:
        filewriter.writerow(line)
