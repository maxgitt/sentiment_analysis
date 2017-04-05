The purpose of this project was to understand how to use various classifiers for classification of a real world problem. The first part of the project focused on lemmatization, stemming, and removal of stopwords to produce both a term-frequency (TF) feature matrix and a term frequency-inverse document frequency (TF-IDF) feature matrix.

Using scikit learn, various classifiers were tested for highest classification accuracy.


Additional:
    
    A lot of junk data exists in each file that will not be useful in training and predicting a label.
    For example:
        punctuation: , ! . ; :
        break statements: <br/>

    It would be unwise to remove all punctuation at will because of the following example:
        "I J.C. walked!" -> "I J C walked"

    Numbers are also found to be irrelevant because without context a number is objective.
    Therefore I found it best to remove numbers, punctuation, breakpoints, but keep punctuation for acronyms and
    apostrophes for possesive nouns. I check if the a word has atleast 1 alpha character (A-Z or a-z).
        Remove:
            "."
            "<br/>"
            "2/10"
        Keep:
            "U.S."
            "she's"
    The list comprehension to check if a word has atleast 1 alpha character is simpler and significantly faster
    than a regular expression to do the above.

2(c) Model Accuracy:
    Logistic Regression:      C: 0.001   0.01   0.01     1      10    100    1000
                                [0.8916, 0.8841, 0.8724, 0.8688, 0.8656, 0.8528, 0.8424]

    SVM with linear kernel:  C: 0.001   0.01    0.1     1       10    100
                                [0.8829, 0.8688, 0.8636, 0.8596, 0.8712, 0.8774]

    Naive Bayes model: 0.638

2(d) Model accuracy of td Only include best C value and corresponding accuracy:
             best C      highest accuracy
    (i)  A:  0.01             0.8106
         B:  0.001            0.8425

    (ii) A:  0.001            0.8425
         B:  0.1              0.7849

2(e) SVM with linear kernel accuracy
         C:      0.001   0.01    0.1      1       10     100
    Accuracy:  [0.5488, 0.6332, 0.6952, 0.7332, 0.7476, 0.7504]


main.py does not need to re-ouptput the results from sections 2(c), 2(d), or 2(e)
(But the individual functions sec2c and Word2vec should output our results from 2(c) and 2(e)
