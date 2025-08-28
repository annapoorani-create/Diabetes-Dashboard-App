

# imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cleaning up our data
df = pd.read_csv('diabetes (1).csv')
df_temp = df.drop(columns=['Outcome','Pregnancies','Insulin','SkinThickness'])
df_temp = df_temp.replace(0,np.nan)
df = pd.concat([df['Pregnancies'],df['Insulin'],df['SkinThickness'],df_temp,df['Outcome']],axis=1)
df = df.dropna().reset_index(drop=True)

# rename the column names so they have spaces
df = df.rename(columns={'BloodPressure': 'Diastolic Blood Pressure','DiabetesPedigreeFunction': 'Diabetes Pedigree Function'})

# Setting up the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualizations", "ML Model 1", "ML Model 2","Neural Net"])

# Homepage
if page == "Home":
    st.title("Welcome to the Diabetes App!")
    st.write("Use the sidebar to navigate to different sections.")
    st.write("Here's a quick look at the data:")
    st.dataframe(df.head())
    st.dataframe(df.describe())

    with st.expander("Read more about this dataset!"):
        st.write("""This is data collected from a group of Pima Indian women, aged 21 and above. 
        Zeros indicate missing values, except in the case of outcome, where they indicate a non-diabetic person. 
        There are 768 rows of data, and 8 features, not including Outcome.""")

# Data Viz Page
elif page == "Data Visualizations":
    st.title("Let's take a look, shall we?")
    st.write("Choose a figure you're curious about!")

    # Plots we already created!
    if st.button("Display Age Histogram"):
        st.title("Age Histogram")
        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=20)
        st.pyplot(fig)
    if st.button("Display Glucose vs. BMI Scatterplot"):
        st.title("Glucose vs BMI")
        fig, ax = plt.subplots()
        ax.scatter(df["Glucose"], df["BMI"])
        st.pyplot(fig)
    if st.button("Display Correlation Heatmap"):
        corr = df.corr()
        st.title("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data=corr, ax=ax,cmap='coolwarm', annot=True)
        st.pyplot(fig)

        # Correlation Heatmap Explanation
        with st.expander("What is this plot showing?"):
            st.write("""This is a correlation heatmap. Warmer colors indicate positive correlations and cooler colors indicate negative correlations. 
            The closer the absolute value of the correlation is to 1, the more strongly correlated.""")
        
    # Buttons for distribution by outomce
    st.write("Choose a characteristic to see its distribution by outcome.")
    for feature in df.columns:
        if feature == 'Outcome':
            continue
        if st.button(feature):
            fig, ax = plt.subplots()
            sns.kdeplot(data = df[df['Outcome']==1], x = feature, label = 'Diabetic', ax=ax);
            sns.kdeplot(data = df[df['Outcome']==0], x = feature, label = 'Non-diabetic',ax=ax);
            ax.set_title(f'{feature} Distribution by Outcome')
            ax.legend()
            st.pyplot(fig)
        
    
# Logistic Regression Page
elif page == "ML Model 1":

    # imports
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    st.title("Logistic Regression Model")
    # Explain the model
    with st.expander("How does this model work?"):
            st.write("""This is a logistic regression model, often used for classification tasks, such as 'spam' vs. 'not spam' or our scenario, 
            'diabetic' vs. 'non-diabetic'. It multiplies each value for each feature in the information passed to it and sums all of them;  
            it then passes this sum through a sigmoid function, which squeezes this to ensure all values are between 0 and 1.  For larger sums, 
            the output is closer to one, and smaller sums produce an output closer to 0. If the final output is over a certain threshold, usually ~0.5, 
            the model returns '1.' Otherwise, it returns '0.'
            """)

    # Choosing features
    features = st.multiselect("Choose features:", df.columns[:-1], default=["Glucose", "BMI"])
    target = "Outcome"
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

    # Training the model
    if st.button("Train model"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        st.success(f"Accuracy: {acc:.2f}")
        st.write("Confusion matrix:")
        st.write(cm)

        #Saving the model
        st.session_state['model'] = model
        st.session_state['features'] = features

    # Show form for prediction if model exists
    if 'model' in st.session_state and 'features' in st.session_state:
        st.subheader("Make a Prediction")

        #Reload the model
        model = st.session_state['model']
        features = st.session_state['features']

        # Allow users to enter data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in features:
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0)
            submitted = st.form_submit_button("Predict!")
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")

elif page == "ML Model 2":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    st.title("Decision Tree Classifier")
    with st.expander("How does this model work?"):
            st.write("""A decision tree creates a series of splits, or branches, using layers of condtions such as 'Age < 30' and 'Age > 30' to classify data. When making a prediction, it simply follows the path of the tree, given a certain piece of data, and outputs the result of the last branch it reaches, which in our case is either a 0 or a 1.""")
            
    # Allow user to select features
    features = st.multiselect("Choose features:", df.columns[:-1], default=["Glucose", "BMI"])
    target = "Outcome"
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

    # Train model
    if st.button("Train model"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        model1 = DecisionTreeClassifier()
        model1.fit(X_train, y_train)

        preds = model1.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        st.success(f"Accuracy: {acc:.2f}")
        st.write("Confusion matrix:")
        st.write(cm)

        with st.expander("What is this showing?"):
            st.write("""This is a confusion matrix, a way to get a quick look at how our model performed on the testing set. The top indicates the actual correct answer - 0 or 1. 
            The first row includes all the model's false predictions; for example, the top left square indicates how many false cases were assigned 'false' by the model. Similarly, the botton 
            row indicates all the model's true predictions, so the bottom left square indicates how many negative cases were assigned 'positive' by the model.""")
            
        # Save model
        st.session_state['model1'] = model1
        st.session_state['features'] = features

    # Show form for prediction if model exists
    if 'model1' in st.session_state and 'features' in st.session_state:
        st.subheader("Make a Prediction")

        # Reload model
        model1 = st.session_state['model1']
        features = st.session_state['features']

        # Allow user to input data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in features:
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0)
            submitted = st.form_submit_button("Predict!")

            # Make a prediction
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                prediction = model1.predict(input_df)
                st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")

if page == "Neural Net":
    # Some Imports
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    

    # More Imports
    import numpy as np
    import pandas as pd

    # Read the CSV
    df = pd.read_csv('diabetes (1).csv')
    df.head()

    # Scale all the data
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Load your data
    df = pd.read_csv('diabetes (1).csv')
    
    # Suppose the last column is the target, so separate features and target
    X = df.iloc[:, :-1]   # all columns except last
    y = df.iloc[:, -1]    # last column
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on features and transform
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame if you want
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Full Scaled Dataframe
    full_scaled_df = pd.concat([X_scaled_df,df['Outcome']],axis=1)
    list_of_lists = full_scaled_df.values.tolist()
    full_scaled_df.head()
    
    # The scaling variables
    my_mean = scaler.mean_
    my_scale = scaler.scale_

    # Build the dataset
    def build_dataset(big_list):  
      X, y = [], []
      for w in big_list:
    
          X.append(w[0:8])
          y.append(w[8])
          
      X = torch.tensor(X)
      y = torch.tensor(y)
      print(X.shape, y.shape)
      return X, y
        
    # Split the datasets
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

    import random
    random.seed(42)
    random.shuffle(list_of_lists)
    n1 = int(0.3*len(list_of_lists))
    n2 = int((1-test_size)*len(list_of_lists))
    
    Xtr, Ytr = build_dataset(list_of_lists[:n1])
    Xdev, Ydev = build_dataset(list_of_lists[n1:n2])
    Xte, Yte = build_dataset(list_of_lists[n2:])
    Ytr = Ytr.long()
    Ydev = Ydev.long()
    Yte = Yte.long()

    # Knobs
    vector_dimensionality_for_features = 8
    hidden_layer_dimension = st.slider("Hidden layer dimension", 10, 30, 20)
    batch_size = st.slider("Batch size for processing", 20, 70, 40)
    my_lr = st.slider("Learning rate", 0.0001, 0.005, 0.001)
    number_of_iterations_of_backprop = st.slider("Number of training loops",1000,10000,5000)

    if st.button("Train Model"):
        # Setting up weights and
        g = torch.Generator().manual_seed(2147483647)
        W1 = torch.randn((vector_dimensionality_for_features,hidden_layer_dimension), generator = g, requires_grad = True)
        with torch.no_grad():
            W1 *= (2.0 / vector_dimensionality_for_features)**0.5
        b1 = torch.randn(hidden_layer_dimension, generator = g, requires_grad = True)
        with torch.no_grad():
            b1 *= 0.01
        W2 = torch.randn(hidden_layer_dimension, 2, requires_grad=True)
        with torch.no_grad():
            W2 *= (2.0 / hidden_layer_dimension)**0.5
        b2 = torch.zeros(2, dtype=torch.float32, requires_grad = True)
        parameters = [W1, b1, W2, b2]
        
        # Optimizer and Scheduler
        optimizer = torch.optim.Adam(parameters, lr=my_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # Because you want to minimize validation loss
            factor=0.5,          # LR will be multiplied by this factor when triggered
            patience=5,        # Number of iterations with no improvement before reducing LR
            # Could make these sliders later
        )

        for i in range(number_of_iterations_of_backprop):
            # mini batch construct
            ix = torch.randint(0,Xtr.shape[0],(batch_size,))
            emb = Xtr[ix]
            
            # forward pass
            hpreact = emb @ W1 + b1
            h = torch.nn.functional.leaky_relu(hpreact, negative_slope=0.01)
            h = torch.nn.functional.dropout(h, p=0.3, training=True)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Ytr[ix])
            
            # (occasionally) print loss
            if i%500 == 0:
                st.write(loss)
                
            # backward pass
            for p in parameters:
                p.grad = None
            # backward pass + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Loss for dev set
            if i%500 == 0:
                emb_for_dev = Xdev
                hpreact_for_dev = emb_for_dev @ W1 + b1
                h_for_dev = torch.nn.functional.leaky_relu(hpreact_for_dev, negative_slope=0.01)
                logits_for_dev = h_for_dev @ W2 + b2
                loss_for_dev = F.cross_entropy(logits_for_dev, Ydev)
                st.write('loss for dev is ',loss_for_dev)
                scheduler.step(loss_for_dev)
        
        st.write(loss)

        # Metrics
        def find_loss_model(sampleX,sampleY):
            # Loss for test and val sets
            emb_for_sample = sampleX
            #forward pass
            hpreact_for_sample = emb_for_sample @ W1 + b1
            h_for_sample = torch.nn.functional.leaky_relu(hpreact_for_sample, negative_slope=0.01)
            logits_for_sample = h_for_sample @ W2 + b2
            loss_for_sample = F.cross_entropy(logits_for_sample, sampleY)
            return logits_for_sample

        with torch.no_grad():
            logits_dev = find_loss_model(Xdev,Ydev)
            preds = torch.argmax(logits_dev, dim=1)
            acc = (preds == Ydev).float().mean()
            st.write(f'Dev accuracy: {acc.item()*100:.2f}%')

        def model2(sampleX):
            emb_for_sample = sampleX
            #forward pass
            hpreact_for_sample = emb_for_sample @ W1 + b1
            h_for_sample = torch.nn.functional.leaky_relu(hpreact_for_sample, negative_slope=0.01)
            logits_for_sample = h_for_sample @ W2 + b2
            return logits_for_sample
        
        # Save model
        st.session_state['model2'] = model2

    # Show form for prediction if model exists
    if 'model2' in st.session_state:
        st.subheader("Make a Prediction")

        # Reload model
        model2 = st.session_state['model2']

        # Allow user to input data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in df.columns:
                if feature == "Outcome":
                    continue
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0)
                
            submitted = st.form_submit_button("Predict!")

            # Make a prediction
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                x_row_scaled = (input_df - my_mean) / my_scale
                logits = model2(torch.tensor(x_row_scaled.values, dtype=torch.float32))
                pred_class = torch.argmax(logits, dim=1).item()
                st.success(f"Prediction: {'Diabetic' if pred_class == 1 else 'Not Diabetic'}")

        
        
        
        
