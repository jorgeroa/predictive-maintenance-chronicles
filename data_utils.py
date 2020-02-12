import re 
import pickle          


# ############# PARAMETERS FOR SAVING/LOADING DATA #################

# Parameters for current data used for training/predition
fold_current_input_data = "pdm/data/input"
fold_current_output_data = "pdm/data/output"

f_current_seq_normal_bin = fold_current_input_data+"/sequences.pick"
f_current_seq_disturbed_bin = fold_current_input_data+"/disturbed_sequences.pick"

f_current_result = fold_current_output_data+"result.txt"
f_current_result_disturbed = fold_current_output_data+"result_disturbed.txt"

# Parameters for generating experiments of data generation

iexp = 4   # The index of the experiment (generator) to be performed
iseq = 1 # The index of the sequence to be generated. N sequences can be generated for one experiment
fold_gen = "experiments/generator{0}".format(iexp)
fold_seq = "/seq{0}".format(iseq)

# #### Binary files #### 
f_generator_bin = fold_gen+"/generator.pick"
f_chronicles_bin = fold_gen+"/chronicles.pick"
f_noisy_chronicles_bin = fold_gen+"/noisy_chronicles.pick"
f_disturbed_chronicles_bin = fold_gen+"/disturbed_chronicles.pick"

f_seq_normal_bin = fold_gen+fold_seq+"/normal.pick"
f_seq_noisy_bin = fold_gen+fold_seq+"/noisy.pick"
f_seq_disturbed_bin = fold_gen+fold_seq+"/disturbed.pick"

# ####  Text files #### 
f_seq_normal_txt = fold_gen+fold_seq+"/normal.txt"
f_seq_noisy_txt = fold_gen+fold_seq+"/noisy.txt"
f_seq_disturbed_txt = fold_gen+fold_seq+"/disturbed.txt"
f_seq_normal_time_txt = fold_gen+fold_seq+"/normal_time.txt"
f_seq_noisy_time_txt = fold_gen+fold_seq+"/noisy_time.txt"
f_seq_disturbed_time_txt = fold_gen+fold_seq+"/disturbed_time.txt"

# #### CSV files #### 
f_seq_normal_csv = fold_gen+fold_seq+"/normal.csv"
f_seq_noisy_csv = fold_gen+fold_seq+"/noisy.csv"
f_seq_disturbed_csv = fold_gen+fold_seq+"/disturbed.csv"

# #### JSON file for configuration of the generation #### 
f_config_generation = fold_gen+fold_seq+"/config_generator.json"

# ############# PARAMETERS FOR SAVING/LOADING MODELS #################
# Current neural network model, parameters, history, and summary used for executing PdM
fold_current_model = "pdm/model"
fold_current_model_summary = fold_current_model + "/summary"
f_current_model = fold_current_model + "/model.h5"
f_current_config = fold_current_model + "/config.pick"
f_current_history = fold_current_model + "/history.pick"

# Parameters for generating experiments of different neural network models
imodel = 1
# fold_model = "pdm/experiments/model{0}".format(imodel)
# fold_model_summary = "pdm/experiments/model{0}/summary".format(imodel)
# fold_input_data = "pdm/experiments/model{0}/data/input".format(imodel)
# fold_output_data = "pdm/experiments/model{0}/data/input".format(imodel)
# f_model = fold_model + "/model.h5"
# f_config = fold_model + "/config.pick"
# f_history = fold_model + "/history.pick"
fold_model = fold_gen+fold_seq+"/model{0}".format(imodel)
fold_model_summary = fold_gen+fold_seq+"/model{0}/summary".format(imodel)
fold_input_data = fold_gen+fold_seq+"/model{0}/input".format(imodel)
fold_output_data = fold_gen+fold_seq+"/model{0}/output".format(imodel)
f_model = fold_gen+fold_seq+"/model{0}/model.h5".format(imodel)
f_config = fold_gen+fold_seq+"/model{0}/config.pick".format(imodel)
f_history = fold_gen+fold_seq+"/model{0}/history.pick".format(imodel)


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

def metrics_to_csv(rows,filename):
    with open(filename, "w+") as f:
        f.write('\"Lambda\",\"Threshold\",\"Accuracy\",\"Precision\",\"Recall\",\"F1\",\"Kappa\",\"AUC\",\"AnomPrecision\",\"AnomRecall\",\"AnomF1\n')
        i=1
        for row in rows: 
            if len(row)==0:
                continue
            f.write(str(row[0])+','+str(row[1])+','+str(row[2])+
                    ','+str(row[3])+','+str(row[4])+','+str(row[5])+
                    ','+str(row[6])+','+str(row[7])+','+str(row[8])+
                    ','+str(row[9])+','+str(row[10])+'\n')
            i=i+1

def scores_to_csv(rows,filename):
    with open(filename, "w+") as f:
        f.write('\"Lambda\",\"Threshold\",\"F1\"\n')
        i=1
        for row in rows: #the rows are "Lambda, Threshold, F1-score"
            if len(row)==0:
                continue
            f.write(str(row[0])+','+str(row[1])+','+str(row[2])+'\n')
            i=i+1

