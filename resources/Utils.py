import pickle


def pretty_print_df(dataframe, head=True):
    if head:
        return dataframe.head(5).to_markdown()
    return dataframe.to_markdown()


def store_object(object, filename):
    file = open(filename, "wb")
    pickle.dump(object, file)
    print("Successfully stored to {}".format(filename))


def load_object(filename):
    file = open(filename, "rb")
    object = pickle.load(file)
    print("Successfully loaded Object from {}".format(filename))
    return object
