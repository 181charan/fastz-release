#!/usr/bin/python

import sys, getopt

class Maf:
     def __init__(self,score,anchor1, al1, tlen1, seq1, anchor2, al2, tlen2, seq2):
         self.score = score
         self.anchor1 = anchor1
         self.al1 = al1
         self.tlen1 = tlen1
         self.seq1 = seq1
         self.anchor2 = anchor2
         self.al2 = al2
         self.tlen2 = tlen2
         self.seq2 = seq2

def main(argv):
   inputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
   except getopt.GetoptError:
      print ('postprocess.py -i <inputfile>.maf ')
      sys.exit(2)
   if ((len(opts) < 1)):
        print ('postprocess.py -i <inputfile>.maf ')
        sys.exit(2)   
   for opt, arg in opts:
      if opt == '-h':
         print ('postprocess.py -i <inputfile>.maf ')
         sys.exit(2)
      elif opt in ("-i", "--ifile"):
         inputfile = arg
   return inputfile

if __name__ == "__main__":
    inputFile = main(sys.argv[1:])
    print(inputFile)
    file = open(inputFile,  'r')
    Lines = file.readlines()
    
    scoreList = []
    chrName = "s chr" # hardcoding this might not be a good idea
    count = 0
    flag = 0
    ctr = 0
    headers = []
    lineCtr = 0

    name1 = ""
    name2 = ""
    strand = ""
    for line in Lines:
        if(lineCtr < 14):
            headers.append(line)
            #print(line, end="")
            lineCtr += 1
        if "score" in line.strip():
            count += 1
            flag = 1
            score = line.strip().split("=")[1]
        if "s chr" in line.strip():
            if flag == 1:
                flag = 0
                ctr = 0
                name1 = (line.strip().split(" ")[1])
                strand = (line.strip().split(" ")[4])
                for i in range(2,len(line.strip().split(" "))):
                    if line.strip().split(" ")[i].isdigit():
                        if ctr == 0:
                            anchor1 = (line.strip().split(" ")[i])
                        elif ctr == 1:
                            al1 = (line.strip().split(" ")[i])
                        else:
                            tlen1 = (line.strip().split(" ")[i])
                        ctr += 1
                    else:
                        if len(line.strip().split(" ")[i]) > 1 :
                            seq1 = (line.strip().split(" ")[i])

            else:
                ctr = 0
                name2 = (line.strip().split(" ")[1])
                for i in range(2,len(line.strip().split(" "))):
                    if line.strip().split(" ")[i].isdigit():
                        if ctr == 0:
                            anchor2 = (line.strip().split(" ")[i])
                        elif ctr == 1:
                            al2 = (line.strip().split(" ")[i])
                        else:
                            tlen2 = (line.strip().split(" ")[i])
                        ctr += 1
                    else:
                        if len(line.strip().split(" ")[i]) > 1 :
                            seq2 = (line.strip().split(" ")[i])
                            scoreList.append( Maf(score,anchor1, al1, tlen1, seq1, anchor2, al2, tlen2, seq2) )
    filList = []
    prevMaf = scoreList[0]
    temp = Maf(score, anchor1, al1, tlen1, seq1, anchor2, al2, tlen2, seq2)
    for s in scoreList:
        if (prevMaf.anchor1 == s.anchor1) and (prevMaf.anchor2 == s.anchor2):
            if (prevMaf.score < s.score):
                prevMaf = s
        else:
            filList.append(prevMaf)
            prevMaf = s
            
    original_stdout = sys.stdout # Save a reference to the original standard output
        
    with open(inputFile, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        for l in headers:
            print(l,end="")
        for s in filList:
            print("a score="+str(s.score))
            temp = s.seq1.replace("-","")
            if(len(temp) != s.al1):
                s.al1 = len(temp)
            line_new = '{} {} {:>8} {:>5} {} {:>8} {}'.format("s", name1 , s.anchor1 , s.al1 , strand, s.tlen1, s.seq1)
            print(line_new)
            temp = s.seq2.replace("-","")
            if(len(temp) != s.al2):
                s.al2 = len(temp)
            line_new = '{} {:>} {:>8} {:>5} {} {:>8} {}'.format("s", name2 , s.anchor2 , s.al2 , strand, s.tlen2, s.seq2)
            print(line_new)
            print()
        sys.stdout = original_stdout # Reset the standard output to its original value
