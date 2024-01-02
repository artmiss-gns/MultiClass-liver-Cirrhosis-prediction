import pickle 

def save_model(model, name="best_model.pkl") :
    # storing the model for later usages
    with open(name, 'wb') as file :
        pickle.dump(model, file)


def load_model(name="best_model.pkl") :
    with open(name, 'rb') as file :
        model = pickle.load(file)
        return model