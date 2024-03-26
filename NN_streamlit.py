import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import random
from sklearn import datasets
import re

# Author: Aditya Koshti, IIT Goa 
# Mentor: Dr. Neha Karanjkar, IIT Goa 


#####################################################
# Define the functions to generate random data
#####################################################

def compute_function_values(df, num_rows, num_axes, num_functions):
    f_names=r"" 
    for i in range(num_functions):
        # make a random array of coefficients and powers to create each function as a random polynomial
        f_names+="\n - $$f_"+str(i+1)+"="        
        powers = [random.choice([0,1,2,3]) for i in range(num_axes)]
        coeffs = [random.randint(-3,3) for i in range(num_axes)]
        function_column = np.zeros(num_rows)
        for j in range(num_axes):
            function_column += (coeffs[j] * df['x' + str(j + 1)].pow(powers[j]))
            f_names+=str(abs(coeffs[j]))+" x_"+str(j+1)+"^{"+str(powers[j])+"}"
            if j != (num_axes-1):
                f_names+="+" if (coeffs[j+1] >0) else "-"
        f_names+="$$"
        df['f' + str(i + 1)] = function_column
    return df


def generate_random_data(num_rows, num_axes, num_functions):
    axis_points = np.random.random(size=(num_rows, num_axes))
    df = pd.DataFrame(axis_points, columns=['x' + str(i) for i in range(1, num_axes + 1)])
    return compute_function_values(df, num_rows, num_axes, num_functions)
 

def slice_data(data, other_axes, values):
    data_slice=data
    for i, axis in enumerate(other_axes):
        data_slice = data_slice[(data_slice[axis] >= values[i][0]) & (data_slice[axis] <= values[i][1])]
    return data_slice

from scipy.interpolate import Rbf

# generate an interpolation from the data using Radial Basis Functions

def interpolate_rbf(df, axis_columns, function_columns):
    rbf_functions=[]
    points = df[axis_columns].to_numpy()
    for function_column in function_columns:
        values = df[function_column].to_numpy()
        rbf_function = Rbf(*points.T, values, function='thin_plate')
        rbf_functions.append(rbf_function)
    return rbf_functions
    

def plot_3d(data_slice, selected_x_axis, selected_y_axis, selected_function):
    fig = px.scatter_3d(data_slice, x=selected_x_axis, y=selected_y_axis,z=selected_function, color=selected_function, color_continuous_scale=st.session_state.colorscale)
    fig.update_layout(scene = dict(aspectmode='cube'),template='plotly',
        margin={"l":0,"r":0,"t":0,"b":0} 
    )
    fig.update_traces(marker_size=st.session_state.marker_size)
    return fig

#####################################################
# Define the neural network
#####################################################
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activations):
        super(CustomNeuralNetwork, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation_mapping[activations[0]]())

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(activation_mapping[activations[i]]())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Activation functions mapping
activation_mapping = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}

# Loss functions mapping
loss_mapping = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "SmoothL1": nn.SmoothL1Loss,
}


# Function to train the neural network
def train(model, x_train, y_train, x_val, y_val, epochs, lr, criterion):
    losses = []
    val_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs = torch.tensor(x_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        with torch.no_grad():
            val_inputs = torch.tensor(x_val, dtype=torch.float32)
            val_targets = torch.tensor(y_val, dtype=torch.float32)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_losses.append(val_loss.item())

        
    return losses, val_losses


def count_variables(math_expression):
    unique_variables = []
    num_variables = []
    # dictionary to store the unique variables and ap it to 1 to n
    var_dict = {}

    for i in range(len(math_expression)):
        # Use regular expression to find variable names
        variable_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', math_expression[i])
        # Count the unique variable names   
        unique_variables.append(set(variable_names))
        num_variables.append(len(set(variable_names)))
        # Map the unique variable names to 1 to n
        for var in unique_variables[i]:
            if var not in var_dict:
                var_dict[var] = len(var_dict) + 1

    return num_variables, unique_variables, var_dict


def eval_function(data, math_expression, x_values, unique_variables, var_dict):
    
    variable_dict = {}
    for var in unique_variables[0]:
        variable_dict[var] = x_values[:, var_dict[var] - 1]
    # Evaluate the math expression
    y_values = np.array(eval(math_expression[0], variable_dict)).reshape(-1, 1)
    data["y1"] = y_values
    for i in range(1, len(math_expression)):
        # Create a dictionary to map variable names to values
        variable_dict = {}
        for var in unique_variables[i]:
            variable_dict[var] = x_values[:, var_dict[var] - 1]
        # Evaluate the math expression
        arr = np.array(eval(math_expression[i], variable_dict)).reshape(-1, 1)
        # print("Eval", arr.shape)
        y_values = np.append(y_values, arr, axis=1)
        data[f"y{i+1}"] = arr

    #print("Eval", y_values.shape)
    return y_values


def create_graph(graph_name, training_points, data, input_size, output_size):
    
    st.session_state.data_generated=True
    ####################################
    # Visualize
    ####################################
    st.subheader(graph_name)
    print("Data")
    print(data)

    if st.session_state.data_generated:
        column_names=list(data.columns)
        st.write("Select columns corresponding to the point coordinates and function values:")
        default_axes = column_names[0:input_size]

        axes = st.multiselect("x and y axis:", column_names, default=default_axes, key=graph_name+"axes")
        default_functions = [f for f in column_names if f not in axes]
        functions = st.multiselect("function values:", default_functions, default=default_functions, key=graph_name+"functions")


        st.write("Select the axises and the function you want to visualize:")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_x_axis = st.selectbox("x-axis", axes, index=0, key=graph_name+"x")
        with col2:
            selected_y_axis = st.selectbox("y-axis", [a for a in axes if (a!=selected_x_axis)], index=0, key=graph_name+"y")
        with col3:
            selected_function = st.selectbox("function component", functions, index=0, key=graph_name+"f")
        
        if selected_x_axis == selected_y_axis or selected_y_axis==selected_function or selected_x_axis == selected_function:
            st.error("Please select two distinct axes to visualize the meta-model")
            

        st.markdown("""---""")
        # Get the values of the other axes from the user
        other_axes = [i for i in axes if i not in [selected_x_axis, selected_y_axis]]
        values = []
        center_values = []
        if other_axes:
            i = 0
            st.write("Select the range of values for the other axes")
            for axis in other_axes:
                min_value = float(data[axis].min())
                max_value = max(float(data[axis].max()), min_value+1)
                val = st.slider(f"{axis} range", min_value=min_value, max_value=max_value, value=(min_value, max_value), key=f"{axis}+{graph_name}")
                values.append(val)
                center_values.append(float(val[0]+val[1])/2)
                i+=1
        
        data_slice = slice_data(data, other_axes, values)
        if(len(data_slice)==0):
            st.warning(f"The selected slice contains {len(data_slice)} points")
        else:
            st.info(f"The selected slice contains {len(data_slice)} points")
        fig = plot_3d(data_slice, selected_x_axis, selected_y_axis, selected_function)
        st.plotly_chart(fig, use_container_width=True)

        return selected_x_axis, selected_y_axis, selected_function

def plot_loss(losses, graph_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(losses)), y=losses, mode='lines+markers', name='loss'))
    st.markdown("""---""")
    st.subheader(graph_name)
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Loss",
        title="Loss vs Epochs"
    )
    st.plotly_chart(fig, use_container_width=True)

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean = np.mean(y_true)
    ss_total = np.sum((y_true - mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2

def mean_squared_error(y_true, y_pred, squared=True):
    mse = np.mean((y_true - y_pred) ** 2)
    if squared:
        return mse
    else:
        return np.sqrt(mse)


def goodness_of_fit(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse


def contour_plot(x_values, y_values, z_values):
    st.title("Contour Plotter")

    try:
        # Generate meshgrid of points
        X, Y = np.meshgrid(x_values, y_values)
        X = x_values
        Y = y_values
        Z = z_values

        # Plot contour graph
        fig = go.Figure(data=[go.Contour(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(title="Contour Plot of Function", autosize=False)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {e}")


def compute_function_values(df, num_rows, num_axes, num_functions):
    f_names=r"" 
    for i in range(num_functions):
        # make a random array of coefficients and powers to create each function as a random polynomial
        f_names+="\n - $$f_"+str(i+1)+"="        
        powers = [random.choice([0,1,2,3]) for i in range(num_axes)]
        coeffs = [random.randint(-3,3) for i in range(num_axes)]
        function_column = np.zeros(num_rows)
        for j in range(num_axes):
            function_column += (coeffs[j] * df['x' + str(j + 1)].pow(powers[j]))
            f_names+=str(abs(coeffs[j]))+" x_"+str(j+1)+"^{"+str(powers[j])+"}"
            if j != (num_axes-1):
                f_names+="+" if (coeffs[j+1] >0) else "-"
        f_names+="$$"
        df['f' + str(i + 1)] = function_column
    return df, f_names

# Write a function which has similar structure as the above function but generates function which have nice plots
def compute_function_values_v2(df, num_rows, num_axes, num_functions):
    f_names=r"" 
    for i in range(num_functions):
        # make a random array of coefficients and powers to create each function as a random polynomial
        f_names+="\n - $$f_"+str(i+1)+"="        
        powers = [random.choice([1,2,3]) for i in range(num_axes)]
        coeffs = [random.randint(-3,3) for i in range(num_axes)]
        function_column = np.zeros(num_rows)
        for j in range(num_axes):
            function_column += (coeffs[j] * df['x' + str(j + 1)].pow(powers[j]))
            f_names+=str(abs(coeffs[j]))+" x_"+str(j+1)+"^{"+str(powers[j])+"}"
            if j != (num_axes-1):
                f_names+="+" if (coeffs[j+1] >0) else "-"
        f_names+="$$"
        df['f' + str(i + 1)] = function_column
    return df, f_names
        
def generate_random_data(num_rows, num_axes, num_functions):
    axis_points = np.random.random(size=(num_rows, num_axes))
    df = pd.DataFrame(axis_points, columns=['x' + str(i) for i in range(1, num_axes + 1)])
    return compute_function_values(df, num_rows, num_axes, num_functions)

# Streamlit UI
def main():

    st.set_page_config(
        page_title="NN Visualiser",
        page_icon="neural-network.png",
        layout="wide",
        initial_sidebar_state="expanded")
    
    c1, c2 = st.columns([20,80])
    with c2:
        st.title("Multi-Dimentional Neural Network Visualizer")
        st.caption("A simple tool to visualize neural-network over multidimentional input/output")
        st.markdown("""
        The Multi-dimensional Neural Network Visualizer is a tool designed to help users understand and analyze the behavior of neural networks in multi-dimensional spaces. It provides an interactive interface to visualize the structure of neural networks, input-output relationships, and model performance metrics.
        """
        )
    with c1:
        st.image("network.png",width=200)


    st.markdown("""---""")
    st.info("Instructions:    \n 1) Choose the Method you want to use to generate functions.  \n 2) Set Neural Network Parameters:  \na) Adjust the Neural Network , Function and Learning Parameters.[In SideBar]  \nb) Below there you will find two graphs for Actual Function and Predicted function from Neural Network.  \n 3) To get new functions click on the Re-Generate button.")
    # st.info("The xi's are the input variables and the yi's are the output variables.")
    # st.info("The variables that you type in function will be termed as x1, x2, etc. in the graph")
    output_func = []
    manual = False

    # Prevent the whole page from rerendering
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False

    c1,c2=st.columns([40,60])
    
    with c1:
        data_choice = st.radio("Select a Method:",  ( 'Generate synthetic data and functions', 'Type-in functions manually'), index=0)

        if data_choice == 'Generate synthetic data and functions':
            num_axes = st.slider("Number of input dimensions (axes)", min_value=1, max_value=10, value=2, step=1)
            num_functions= st.slider("Number of output dimensions (function components)", min_value=1, max_value=10, value=2, step=1)
            num_rows = st.slider("Number of data samples ", min_value=1000, max_value=10000, value=1000, step=100)
            
        
        elif data_choice == 'Type-in functions manually':
            num_axes = st.slider("Number of input dimensions (axes)", min_value=2, max_value=10, value=3, step=1)
            num_functions= st.slider("Number of output dimensions (function components)", min_value=2, max_value=10, value=3, step=1)
            num_rows = st.slider("Number of data samples", min_value=1000, max_value=10000, value=1000, step=100)
            

    with c2:
        
        state_button = st.button(('Re-Generate!' if 'data_generated' in st.session_state else 'Generate!'))
        if state_button:
            st.session_state.data_generated = False
        if state_button or st.session_state.data_generated:
            
            show_func = True
            if data_choice == 'Generate synthetic data and functions' and not st.session_state.data_generated:
                st.session_state.data_generated=True
                data,f_names = generate_random_data(num_rows,num_axes,num_functions)
                data_temp_val, f_names_val = generate_random_data(num_rows, num_axes, num_functions)
                st.session_state.data = data
                st.session_state.data_temp_val = data_temp_val
                st.session_state.f_names = f_names
                st.markdown(f_names)
                show_func = False
            elif not st.session_state.data_generated:
                st.session_state.data_generated=True
                for i in range(num_functions):
                    output_func.append(st.text_input(f"Enter a function {i+1}:", "x+y", key=f"out{i}"))
            
            if show_func:
                st.markdown(st.session_state.f_names)
            # with st.expander("View raw data"):
            #     st.write(data)
            # st.download_button('Download generated data as a CSV file', to_csv(data), 'sample_data.csv', 'text/csv')
    

    if not st.session_state.data_generated :
        return
    

    
    input_size = num_axes
    output_size = num_functions
    num_points = num_rows
    ###########################################
    # User input for neural network parameters
    ###########################################
    # Sidebar
    st.sidebar.header("Neural Network Parameters")

    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 20, 1)

    hidden_layer_sizes = []
    hidden_layer_activations = []

    for i in range(num_hidden_layers):
        hidden_layer_sizes.append(st.sidebar.number_input(f"Hidden Layer {i+1} Size", 1, 100, 10))
        hidden_layer_activations.append(st.sidebar.selectbox(f"Hidden Layer {i+1} Activation", list(activation_mapping.keys())))

    st.sidebar.header("Learning Parameters")
    learning_rate = st.sidebar.number_input("Learning Rate", 0.001, 0.1, step=0.001, format="%.4f")
    epochs = st.sidebar.slider("Epochs", 100, 5000, step=100, value=500)
    loss_function = st.sidebar.selectbox("Loss Function", list(loss_mapping.keys()))

    # User input for function and prediction
    st.sidebar.header("Function Prediction")

    # Percentage of validation points
    val_percent = st.sidebar.slider("Percentage of Test Points", 0, 100, 40)

    with st.sidebar:
        marker_size = st.slider("⚙️ Marker size", min_value=1, max_value=10, value=7, step=1,key="marker_size")
        colorscale = st.selectbox(' ⚙️ Color scale',px.colors.named_colorscales(),key="colorscale", index=1) #use 48 for Virdis


    st.sidebar.markdown("---")
    

    # Build the neural network
    model = CustomNeuralNetwork(input_size, hidden_layer_sizes, output_size, hidden_layer_activations)

    

    training_points = int(num_points * (1 - val_percent/100))
    test_points = num_points - training_points
    print("Training points" , training_points)
    print("Test points", test_points)

    lower = -10
    higher = 10

    

    ###############################################
    # Train the neural network on the function data
    ###############################################

    if 'data' not in st.session_state:

        #check for every function the number of variables
        num_vars, unique_var, var_dict = count_variables(output_func)
        for i in range(len(num_vars)):
            if num_vars[i] > input_size:
                st.error(f"Number of variables in function {i+1} should be less than the input size ({input_size})")
                return

        # Training data
        x_train = np.random.uniform(low=lower, high=higher, size=(training_points, input_size))
        x = x_train
        data1 = pd.DataFrame(x_train, columns=[f"x{i+1}" for i in range(input_size)])
        # print(data1)
        y_train = eval_function(data1, output_func, x_train, unique_var, var_dict)
        # Validation data
        x_val = np.random.uniform(low=lower, high=higher, size=(training_points, input_size))
        data_val = pd.DataFrame(x_val, columns=[f"x{i+1}" for i in range(input_size)])
        y_val = eval_function(data_val, output_func, x_val, unique_var, var_dict)

        losses, val_losses = train(model, x_train, y_train, x_val, y_val, epochs, learning_rate, loss_mapping[loss_function]())

    else :
        data = st.session_state.data
        data_temp_val = st.session_state.data_temp_val
        train_data = data.iloc[:training_points]
        data_temp_val = data_temp_val.iloc[:training_points]
        
        selected_columns =[f"x{i+1}" for i in range(input_size)]
        x_train = train_data[selected_columns].to_numpy()
        selected_columns =[f"f{i+1}" for i in range(output_size)]
        y_train = train_data[selected_columns].to_numpy()
        data1 = data
        selected_columns =[f"x{i+1}" for i in range(input_size)]
        x_val = data_temp_val[selected_columns].to_numpy()
        selected_columns =[f"f{i+1}" for i in range(output_size)]
        y_val = data_temp_val[selected_columns].to_numpy()
        # for i in selected_columns:
        #     data1.drop(columns=[i], inplace=True)
        # x_val = x_train
        # y_val = y_train
        print("x Data")
        print(data1)

        losses, val_losses = train(model, x_train, y_train, x_val, y_val, epochs, learning_rate, loss_mapping[loss_function]())
    
    st.markdown("---")
    st.info("Multidimentional Data Visualiser:  \n --> Select the input variables or dimentions to plot the graph.  \n --> Set the range of other inputs in which you want to visualise the points")
    c1, c2 = st.columns(2)
    with c1:
        # Plot the actual function and predicted function
        x_1, y_1, z_1 = create_graph("Actual Function", training_points, data1, input_size, output_size)
        # Contour Plot of the graph
        
        # contour_plot(data1[x_1], data1[y_1], data1[z_1])

        # Plotting the loss over iterations
        plot_loss(losses, "Training Loss Function")

    
    if 'data' not in st.session_state:
        x_range = np.random.uniform(low=lower, high=higher, size=(test_points, input_size))
        data2 = pd.DataFrame(x_range, columns=[f"x{i+1}" for i in range(input_size)])
        x_tensor = torch.tensor(x_range, dtype=torch.float32)
        predictions = model(x_tensor).detach().numpy()
        # Add the predicted function to the dataframe with y{i}
        for i in range(output_size):
            data2[f"f{i+1}"] = predictions[:, i]
    
    else:
        data = st.session_state.data
        test_data = data.tail(test_points)
        test_data = test_data.reset_index(drop=True)
        selected_columns =[f"x{i+1}" for i in range(input_size)]
        x_range = test_data[selected_columns].to_numpy()
        data2 = test_data
        selected_columns =[f"f{i+1}" for i in range(output_size)]
        for i in selected_columns:
            data2.drop(columns=[i], inplace=True)
        x_tensor = torch.tensor(x_range, dtype=torch.float32)
        predictions = model(x_tensor).detach().numpy()
        # Add the predicted function to the dataframe with y{i}
        for i in range(output_size):
            data2[f"f{i+1}"] = predictions[:, i]

        
    with c2:
        # Plot the actual function and predicted function
        create_graph("Predicted Function", test_points, data2, input_size, output_size)

        # Plotting the loss over iterations
        plot_loss(val_losses, "Validation Loss Function")



    
    # Finding goodness of fit
    st.markdown("""---""")
    st.header("Goodness of Fit")
    st.info("Goodness of fit is a measure of how well the predicted values match the actual values. It is a measure of how well the model fits the data.")


    if 'data' not in st.session_state:

        y_actual_pred = eval_function(data2, output_func, x_range, unique_var, var_dict)
        goodness =  goodness_of_fit(y_actual_pred, predictions)
        st.write("R2 Score: ", goodness[0])

    else:
        data = st.session_state.data
        test_data = data.tail(test_points)
        test_data = test_data.reset_index(drop=True)
        selected_columns =[f"f{i+1}" for i in range(output_size)]
        y_actual_pred = test_data[selected_columns].to_numpy()
        # print("Y_actual_pred", y_actual_pred)
        # print("Predictions", predictions)
        goodness =  goodness_of_fit(y_actual_pred, predictions)
        st.write("R2 Score: ", goodness[0])

    
    st.write("Mean Squared Error: ", goodness[1])
    

if __name__ == "__main__":
    main()

    

# slice 