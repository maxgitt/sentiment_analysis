# input:  a list l of strings
# output: a list containing the lemmatized strings
import math
import numpy as np

def binarysearch(sequence, value):
    lo, hi = 0, len(sequence) - 1
    while lo <= hi:
        mid = (lo + hi) / 2
        if sequence[mid] < value:
            lo = mid + 1
        elif value < sequence[mid]:
            hi = mid - 1
        else:
            return mid
    return -1

def accuracy(prediction, values):
    result = 0
    for p, v in zip(prediction, values):
        if p == v:
            result += 1
    return float(result)/float(len(values))

def lemmatize(l):
    result = []

    from stanford_corenlp_pywrapper import CoreNLP
    proc = CoreNLP("pos", corenlp_jars=["stanford-corenlp-full-2015-04-20/*"], UnicodeDecodeError='skip')

    for doc_words in l:
        single_dict = proc.parse_doc(doc_words)
        row = []
        for each_dict in single_dict['sentences']:
            for word in each_dict['lemmas']:
                row.append(word)
        result.append(row)

    return result


# input:  a list l of strings
# output: a list containing the stemmed strings in l
def stem(l):
    #result = []

    from nltk.stem.porter import PorterStemmer
    p = PorterStemmer()

    result = [[p.stem(word) for word in sentence] for sentence in l]
    return result

# input:  a list l of strings
# output: a list of strings where the stopwords are removed
def removeStopwords(l):
    result = []

    from nltk.corpus import stopwords

    for line in l:
        result.append([word for word in line if word not in stopwords.words('english')])

    return result

# input:  a list l of strings, preprocessed, each i contains all words for train/test/validate
# output: a matrix where the (i,j) component is how many times 
#         the j-th word appear in the i-th document
def tf(l):

    result = [None]*len(l)
    #result = np.array(result)

    #insert all words into the vocabulary
    dupl_vocab = []
    for line in l:
        for word in line:
            dupl_vocab.append(word)

    #sort and remove duplicates
    dupl_vocab.sort()
    dup = dupl_vocab[0]
    vocab = [dup]
    for each in dupl_vocab[1:]:
        if each != dup:
            dup = each
            vocab.append(each)

    for i_idx, i in enumerate(l):
        row = [0]*len(vocab)
        for word in i:
            #check if word exists in vocab
            idx = binarysearch(vocab, word)
            if(idx != -1):
                row[idx] += 1
        result[i_idx] = row

    return vocab, result

# input:  a list l of string
# output: a matrix where the (i,j) component is the tf-idf value of the j-th word in the i-th document
def tfidf(l, tf_matrix, vocab, train_len):

    #convert to np arrays
    result = [None]*train_len
    result = np.array(result)

    idf = [None]*len(vocab)
    idf = np.array(idf)

    #calculate idf for each jth word
    for j_idx, j in enumerate(vocab):
        denom = 0
        for i_idx in range(len(tf_matrix)):
            if tf_matrix[i_idx][j_idx] > 0:
                denom += 1
        idf[j_idx] = math.log(float(len(l))/denom)

    #multiply to calculate tfidf value
    for i_idx in range(train_len):
        result[i_idx] = np.multiply(tf_matrix[i_idx],idf)

    return result.tolist()

# add any additional preprocessing you find helpful
def additional(l):
    result = []

    import re
    #result = [w for w in l if re.search(r'^[a-zA-Z]', w)]
    #only keep words that contain atleast one character A-Z
        #numbers are not relevant, punction is removed, acronyms are kept, breakpoints are removed
    for line in l:
        result.append([w for w in line if w[0].isalpha()])

    #remove breakpoints
    #split string into each word and remove break points
    #add = re.split('\s+', line[3:])
    #train_text.extend([word.lower() for word in add if word != '' and word != "<br" and word != "/>"])

    return result

# input:  a list l of string
# output: a feature matrix like object ready for training (2-D list, numpy array, sparse matrix)
# you may choose to use a subset of the previous functions that work best for you
def preprocess(l, train_len):

    import pickle
    '''
    #preprocess training text
    l = lemmatize(l)
    print "done lemma"

    #d_ii_b
    lemma_words = l

    f = open('lemma_words.txt', 'w')
    pickle.dump(lemma_words, f)
    f.close()
    '''

    print "unload lemma words"
    f = open('lemma_words.txt', 'r')
    lemma_words = pickle.load(f)
    f.close()


    '''
    l = additional(l)
    print "done add"
    l = stem(l)
    print "done stem"

    l = removeStopwords(l)
    print "done remove stopwords"
    '''


    '''
    f = open('remove_stops.txt', 'w')
    pickle.dump(l, f)
    f.close()
    '''

    print "start unloading stopwords"
    f = open('remove_stops.txt', 'r')
    l = pickle.load(f)
    f.close()

    l = l[:train_len/2] + l[7000:7000+train_len/2]

    #when importing lemma words uncomment this
    lemma_words = lemma_words[:train_len/2] + lemma_words[7000:7000+train_len/2]

    print "start tf"

    #call tf (returns a matrix) and tf-idf
    vocab, tf_mat = tf(l)
    vocab_lemma, tf_mat_lemma = tf(lemma_words)
    print "done tf matrix"

    tfidf_matrix = tfidf(l, tf_mat, vocab, train_len)
    tfidf_matrix_lemma = tfidf(lemma_words, tf_mat_lemma, vocab_lemma, train_len)
    print "done tfidf matrix"

    return tfidf_matrix, tf_mat, tfidf_matrix_lemma

# train_X: feature matrix for training
# train_t: list of labels for training
# val_X: feature matrix for validation
# val_t: list of labels for validation
# just print out your results, no need to return any value
def sec2c(train_X, train_t, val_X, val_t, d):

    #list of each c value (hyperparemeter)
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    import numpy as np

    #Logistic Regression
    logistic_accur = [0]*len(C)
    for idx, c in enumerate(C):
        model = LogisticRegression(C=c)
        model.fit(train_X, train_t)
        prediction = model.predict(val_X)

        logistic_accur[idx] = accuracy(prediction, val_t)

    #print accuracy
    print "done logistic"
    print logistic_accur

    if d == True:
        return

    #Linear SVM
    #penalty factor
    Cost = [0.001, 0.01, 0.1, 1, 10, 100]

    svm_acc = [0]*len(Cost)
    for idx, c in enumerate(Cost):
        model = LinearSVC(C=c)
        model.fit(train_X, train_t)
        prediction = model.predict(val_X)

        svm_acc[idx] = accuracy(prediction, val_t)

    #print accuracy
    print "done svm"
    print svm_acc

    #Naive Bayes model
    model = GaussianNB()

    X1 = train_X[:len(train_X)/3]
    X2 = train_X[len(train_X)/3:2*len(train_X)/3]
    X3 = train_X[2*len(train_X)/3:]
    t1 = train_t[:len(train_X)/3]
    t2 = train_t[len(train_X)/3:2*len(train_X)/3]
    t3 = train_t[2*len(train_X)/3:]

    model.partial_fit(X1, t1, classes=np.unique(train_t))
    model.partial_fit(X2, t2, classes=np.unique(train_t))
    model.partial_fit(X3, t3, classes=np.unique(train_t))

    prediction = model.predict(val_X)
    print "done Naive Bayes"
    print accuracy(prediction, val_t)

    return

# input train_text, vali_text, test_text: each being a list of strings
#       train_labels, vali_labels: each being a list of labels
def useWord2vec(train_text, train_labels, vali_text, vali_labels, test_text):

    from sklearn.svm import LinearSVC
    from gensim.models import Word2Vec

    # merge your texts here - unprocessed
    sentences = []
    sentences.extend(train_text)
    sentences.extend(vali_text)
    sentences.extend(test_text)
    sentences = [line.split() for line in sentences]

    # train your word2vec here
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

    #feature matrix for train
    train_mat = [[]]*len(sentences[:len(train_text)])
    for idx, doc in enumerate(sentences[:len(train_text)]):
        row_total = [0.0]*100
        #add each word's vector to row
        for word in doc:
            model[word]
            row_total += model[word]
        train_mat[idx] = row_total/len(doc)
    print "done feature mat"

    #feature matrix for validate
    vali_mat = [[]]*len(sentences[len(train_text):len(train_text)+len(vali_text)])
    for idx, doc in enumerate(sentences[len(train_text):len(train_text)+len(vali_text)]):
        row_total = [0.0]*100
        #add each word's vector to row
        for word in doc:
            row_total += model[word]
        vali_mat[idx] = row_total/len(doc)
    print "done valid mat"

    # train your classifiers here, train SVM with a linear kernel
    #Train Linear SVM with train_matrix and test with validation_matrix
    #penalty factor
    Cost = [0.001, 0.01, 0.1, 1, 10, 100]

    #Validate SVM
    svm_acc = [0]*len(Cost)
    for idx, c in enumerate(Cost):
        model = LinearSVC(C=c)
        model.fit(train_mat, train_labels)
        prediction = model.predict(vali_mat)

        svm_acc[idx] = accuracy(prediction, vali_labels)

    #print accuracy
    print "done 2e"
    print svm_acc

    print "done word2vec svm"

def main():

    #variables
    train_text = []
    train_labels = []
    vali_text = []
    vali_labels = []
    test_text = []

    # read data and extract texts and labels
    train = open('train.txt', 'r')
    test = open('test.txt', 'r')
    validation = open('validation.txt', 'r')

    #count = 0
    for line in train:
        #if count == 5000:
        #    break
        if int(line[0:2]) == -1:
            train_labels.append(-1)
        elif int(line[0:2]) == 1:
            train_labels.append(1)
        else:
            print("Error")
            exit(1)
        #count += 1
        train_text.append(line[3:].decode('utf-8').lower())

    #count = 0
    for line in validation:
        #if count == 5000:
        #    break
        if int(line[0:2]) == -1:
            vali_labels.append(-1)
        elif int(line[0:2]) == 1:
            vali_labels.append(1)
        else:
            print("Error")
            exit(1)
        #count += 1
        vali_text.append(line[3:].decode('utf-8').lower())

    #count = 0
    for line in test:
        #if count == 5000:
        #    break
        test_text.append(line.decode('utf-8').lower())
        #count += 1

    train.close()
    test.close()
    validation.close()

    #create vocab list
    vocab_docs = []
    vocab_docs.extend(train_text)
    vocab_docs.extend(vali_text)
    vocab_docs.extend(test_text)

    # do preprocessing
    #vocab_matrix, tf_mat, tfidf_matrix_lemma = preprocess(vocab_docs, len(train_text) + len(vali_text))
    print "done preprocess"

    # train the model - 2(c)
    '''
    train_X = vocab_matrix[:len(train_text)]
    val_X = vocab_matrix[len(train_text):len(train_text)+len(vali_text)]
    sec2c(train_X, train_labels, val_X, vali_labels, False)
    print "done 2c"
    '''

    #word2Vec
    useWord2vec(train_text, train_labels, vali_text, vali_labels, test_text)

    #train the model - 2(d)
    #i A
    '''
    train_X = tf_mat[:len(train_text)]
    val_X = tf_mat[len(train_text):len(train_text)+len(vali_text)]
    sec2c(train_X, train_labels, val_X, vali_labels, True)
    print "done 2ia"
    #i B same as before

    #ii A same as before (same as iB)

    #ii B
    train_X = tfidf_matrix_lemma[:len(train_text)]
    val_X = tfidf_matrix_lemma[len(train_text):len(train_text)+len(vali_text)]
    sec2c(train_X, train_labels, val_X, vali_labels, True)
    print "done 2iib"

    # make predictions and save them
    #USE BEST MODEL
    #train_X = vocab_matrix[:len(train_text)]
    #val_X = vocab_matrix[len(train_text):len(train_text)+len(vali_text)]
    #sec2c(train_X, train_labels, val_X, vali_labels)
    '''
    print "end main"

if __name__ == '__main__':
    main()





