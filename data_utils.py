import re 
import pickle          


# ############# PARAMETERS FOR SAVING/LOADING DATA OF EXPERIMENTS #################
iexp = 1   # The index of the experiment to be performed
iseq = 1 # The index of the sequence to be generated. N sequences can be generated for one experiment
fold_gen = "experiments/generator{0}"

# Binary files
f_generator_bin = fold_gen.format(iexp)+"/generator.pick"
f_chronicles_bin = fold_gen.format(iexp)+"/chronicles.pick"
f_disturbed_chronicles_bin = fold_gen.format(iexp)+"/disturbed_chronicles.pick"

f_seq_normal_bin = fold_gen.format(iexp)+"/sequences/seq{0}_normal.pick".format(iseq)
f_seq_disturbed_bin = fold_gen.format(iexp)+"/sequences/seq{0}_disturbed.pick".format(iseq)

# Text files
f_seq_normal_txt = fold_gen.format(iexp)+"/sequences/seq{0}_normal.txt".format(iseq)
f_seq_disturbed_txt = fold_gen.format(iexp)+"/sequences/seq{0}_disturbed.txt".format(iseq)
f_seq_normal_time_txt = fold_gen.format(iexp)+"/sequences/seq{0}_normal_time.txt".format(iseq)
f_seq_disturbed_time_txt = fold_gen.format(iexp)+"/sequences/seq{0}_disturbed_time.txt".format(iseq)

# CSV files
f_seq_normal_csv = fold_gen.format(iexp)+"/sequences/seq{0}_normal.csv".format(iseq)

# Neural network model
imodel = 1
fold_model = "pdm/experiments/model{0}".format(imodel)
f_model = fold_model + "/model{0}.h5".format(imodel)
f_config = fold_model + "/config{0}.pick".format(imodel)
f_history = fold_model + "/history{0}.pick".format(imodel)

def filterseq(DBseq):
    """
    @JGT: Removes from DBseq all events equal to -1 and empty sequences
    """
    seqs=[]
    for seq in DBseq:
        #seqs.append([(t,e) for (t,e) in seq if e in c.sequence])
        seqs.append([(t,e) for (t,e) in seq if e != -1])
    return [s for s in seqs if len(s)>1]


def deserialization(input):  
     # \d+ is a regular expression which means 
     # one or more digit 
     # output will be like ['100','564','365'] 
    numbers = re.findall('\d+',input) 

     # now we need to convert each number into integer 
     # int(string) converts string into integer 
    numbers = list(map(int,numbers))
    l=[]
    for i in range(0,len(numbers)-1,2):
        l.append((numbers[i],numbers[i+1]))
    return l

def read_text_file(filename):
    print('Reading file ' + filename + "...")
    with open(filename, "r", encoding='utf8') as textfile:
        L = []
        for line in textfile:
            L.append(line.strip())
        print('File contains ', len(L), "lines.\n")
        return L

def save_text_file(seqs,filename):
     with open(filename, "w+") as fichier:
        for seq in seqs:
            fichier.write(str(seq)+'\n')

def serialization(seqs,filename):
     with open(filename, "w+") as fichier:
        for seq in seqs:
            fichier.write(str(seq).strip('[]')+'\n')
            
def serialization2(seqs,filename):
     with open(filename, "w+") as fichier:
        for seq in seqs:
            s=[i[1] for i in seq ]
            fichier.write(str(s).strip('[]')+'\n')

def saveobj(filename,obj):
    with open(filename, 'wb') as config_file:
        pickle.dump(obj, config_file)
        
def loadobj(filename):
    with open(filename, 'rb') as config_file:
        return pickle.load(config_file)

def to_csv(seqs,filename):
    with open(filename, "w+") as f:
        # f.write('\"Id\",\"Time\",\"Event\"\n')
        f.write('\"Time\",\"Event\"\n')
        i=1
        for seq in seqs: #the rows are "ChID,sequence,TC"
            if len(seq)==0:
                continue
            for t,e in seq:
                # f.write(str(i)+','+str(t)+','+str(e)+'\n')
                f.write(str(t)+','+str(e)+'\n')
                i=i+1


