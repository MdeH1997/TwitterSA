import pandas as pd

path = #insert your path

#input file    
snscrapefile =  path + "\\filename_tweetdata.json"#insert your SNScrape filename
reader = pd.read_json(snscrapefile,orient='records', typ='frame',lines=True, chunksize = 750000, nrows = 3750000)
count = 0

#select ids and convert to json
for chunk in reader:
    count += 750000
    tweet_dict = chunk.to_dict()
    twid=chunk['id']
    tweetid_list = chunk['id'].tolist()
    print(len(tweetid_list))
    outfile = path + "\\filename" + str(count) + ".txt"#insert your tweet id list filename
    with open(outfile, 'w') as output:
        for row in tweetid_list:
            output.write(str(row)+'\n')
