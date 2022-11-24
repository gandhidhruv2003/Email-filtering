import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
def email_spam_filter(email):
    df = pd.read_csv("spam.csv")

    le = LabelEncoder()
    df["Spam"] = le.fit_transform(df["Category"])

    X_train, X_test, Y_train, Y_test = train_test_split(df.Message, df.Spam, test_size=0.1)

    vectorizer = CountVectorizer()
    X_train_count = vectorizer.fit_transform(X_train.values)
    X_train_count.toarray()[:3]

    model = MultinomialNB()
    model.fit(X_train_count, Y_train)

    email_count = vectorizer.transform(email)
    for i in model.predict(email_count):
        if i == 0:
            print("Not spam")
        else:
            print("Spam")

    X_test_count = vectorizer.transform(X_test)
    print("The model is ", (model.score(X_test_count, Y_test))*100 ,"% correct")

email = [input("Enter an email message: ")]
email_spam_filter(email)