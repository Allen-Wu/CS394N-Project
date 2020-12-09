import csv
import sys

def csv_preprocess(old_csv_file, new_csv_file):
    new_lines = []
    # Count vocabulary size
    # Map from raw input integer into vocabulary index
    reddit_cnt_voc = {}
    embed_text_voc = {}
    # Placeholder
    start_token = -1
    end_token = -1
    reddit_vec_len = -1
    embed_text_len = -1

    with open(old_csv_file, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            reddit = row[0][1:-1].replace("'", "").strip().split(',')
            reddit = list(map(int, reddit))
            for x in reddit:
                if x not in reddit_cnt_voc:
                    reddit_cnt_voc[x] = len(reddit_cnt_voc)

            embed_text = row[1][1:-1].replace("'", "").strip().split(',')
            embed_text = list(map(int, embed_text))
            for w in embed_text:
                if w not in embed_text_voc:
                    embed_text_voc[w] = len(embed_text_voc)

        start_token = max(list(embed_text_voc.keys())) + 1
        end_token = start_token + 1
        embed_text_voc[start_token] = len(embed_text_voc)
        embed_text_voc[end_token] = len(embed_text_voc)
    csvfile.close()

    with open(old_csv_file, "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            reddit = row[0][1:-1].replace("'", "").strip().split(',')
            reddit = list(map(int, reddit))
            reddit_vec_len = len(reddit)
            embed_text = row[1][1:-1].replace("'", "").strip().split(',')
            embed_text = list(map(int, embed_text))
            embed_text_len = len(embed_text) + 2
            line = reddit + [start_token] + embed_text + [end_token]
            new_lines.append(line)
    csvfile.close()

    with open(new_csv_file, "w") as csvfile:
        filewriter = csv.writer(csvfile)
        for line in new_lines:
            filewriter.writerow(line)
    csvfile.close()

    return reddit_cnt_voc, embed_text_voc, start_token, end_token, reddit_vec_len, embed_text_len
