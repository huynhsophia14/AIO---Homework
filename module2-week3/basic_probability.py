########################
# Create data
########################
import numpy as np

def create_train_data():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'no'],
        ['Sunny', 'Hot', 'High', 'Strong', 'no'],
        ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'no'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'yes']
    ]
    return np.array(data)

train_data = create_train_data()
print(train_data)

#compute prior

def compute_prior_probability(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    # your code here ******************

    for i in range(len(prior_probability)):
        prior_probability[i] = np.sum(train_data[:, -1] == y_unique[i])

    prior_probability = prior_probability/(np.sum(prior_probability))
    
    return prior_probability


prior_probability = compute_prior_probability(train_data)
print("P(play tennis = No) =", prior_probability[0])
print("P(play tennis = Yes) =", prior_probability[1])

#compute likelihood

def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range (0, train_data.shape[1]-1):
        x_unique = np.unique(train_data[:, i]) #'np.unique'removes duplicate values and sorts the unique values in ascending order
        list_x_name.append(x_unique)
        
        x_conditional_probability = []  # Initialize an empty list
        
        for x_val in x_unique:
            probabilities = []  # Initialize an empty list for each unique x_val
            for y_val in y_unique:
                num = np.sum((train_data[:, i] == x_val) & (train_data[:,-1] == y_val))
                den = np.sum((train_data[:,-1] == y_val))
                probabilities.append(num / den if den != 0 else 0)
            
            x_conditional_probability.append(probabilities)
        
        conditional_probability.append(x_conditional_probability)

    return conditional_probability, list_x_name
  
train_data = create_train_data()
likelihood_value, list_x_name = compute_conditional_probability(train_data)
print ("x1 = ", list_x_name[0])
print ("x2 = ", list_x_name[1])
print ("x3 = ", list_x_name[2])
print ("x4 = ", list_x_name[3])

print ("l1 = ", likelihood_value[0])
print ("l2 = ", likelihood_value[1])
print ("l3 = ", likelihood_value[2])
print ("l4 = ", likelihood_value[3])

# This function is used to return the index of the feature name
def get_index_from_value(feature_name, list_features):
    return np.nonzero(list_features == feature_name)[0][0]
train_data = create_train_data()
_ , list_x_name = compute_conditional_probability(train_data)
outlook = list_x_name[0]

i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)

print(i1, i2, i3)
train_data = create_train_data()
conditional_probability , list_x_name = compute_conditional_probability(train_data)

# Compute P(" Outlook "=" Sunny "| Play Tennis "=" Yes ")
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P('Outlook' = 'Sunny'| 'Play Tennis' = 'Yes') = ", np.round(conditional_probability[0][x1][1], 2))
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P('Outlook' = 'Sunny'| 'Play Tennis' = 'No') = ", np.round(conditional_probability[0][x1][0], 2))

##########################
# Train Naive Bayes Model
# ##########################

def train_naive_bayes(train_data):
# Step 1: Calculate Prior Probability
    y_unique = ['no', 'yes']
    prior_probability = compute_prior_probability(train_data)

# Step 2: Calculate Conditional Probability
    conditional_probability , list_x_name = compute_conditional_probability(train_data)

    return prior_probability, conditional_probability, list_x_name
  
###################
# Prediction
# ###################
def prediction_play_tennis(x, list_x_name, prior_probability, conditional_probability):

    joint_probability_0 = 1
    joint_probability_1 = 1

    xi_index = []
    likelihood_x_0 = []
    likelihood_x_1 = []

    for i in range(len(list_x_name)):
        xi_index.append(get_index_from_value(x[i], list_x_name[i]))
        likelihood_x_0.append(np.round(conditional_probability[i][xi_index[i]][0], 2))
        joint_probability_0 = joint_probability_0*likelihood_x_0[i]

        likelihood_x_1.append(np.round(conditional_probability[i][xi_index[i]][1], 2))
        joint_probability_1 = joint_probability_1*likelihood_x_1[i]

    p0 = joint_probability_0*prior_probability[0]
    p1 = joint_probability_1*prior_probability[1]

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1
    return y_pred, p0
  
X = ['Sunny', 'Cool', 'High', 'Strong']
data = create_train_data ()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(data)
pred, p0 = prediction_play_tennis (X, list_x_name, prior_probability, conditional_probability)

print(pred)
print(p0)

if (pred):
    print ("Ad should go!")
else :
    print ("Ad should not go!")
